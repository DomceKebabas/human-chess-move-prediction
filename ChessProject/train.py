"""
Train a move-prediction model (models.py) on the TRAIN split, with early
stopping on the VALIDATION loss, then report val accuracy per band.

Models
  --model linear     Baseline 3: logistic regression on the flat board
  --model mlp        optional baseline: one hidden layer (--hidden)
  --model cnn        main model (--channels, --blocks)
  --no-rating        same model, but the rating input is always 0
                     (the "without rating" half of the core experiment)

Outputs (names default to <model>_rating / <model>_norating, or --name)
  checkpoints/<name>.pt      best model (lowest val loss) + its config
  results_<name>_val.csv     per-band table, same columns as every baseline

Examples
  Baseline 3 on the laptop (100k train positions, CPU is fine):
    python train.py dataset_all.npz --model linear --train-limit 100000
    python train.py dataset_all.npz --model linear --train-limit 100000 --no-rating
  Main model (GPU, e.g. Colab):
    python train.py dataset_all.npz --model cnn
    python train.py dataset_all.npz --model cnn --no-rating
"""
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import legal_cache
from common import add_logging_args, load_dataset, setup_logging
from evaluate_model import pick_device, predict_topk, result_rows
from evaluation import report
from models import build_model, count_parameters, masked_logits
from move_batches import MoveBatches
from splits import TRAIN, VAL

log = logging.getLogger(__name__)


def choose_rows(split, code, limit, rng):
    """Rows of one split; a random subset of `limit` rows if limit > 0."""
    rows = np.flatnonzero(split == code)
    if limit and limit < len(rows):
        rows = np.sort(rng.choice(rows, size=limit, replace=False))
    return rows


def train_one_epoch(model, optimizer, batches, rows, batch_size, rng, device, log_every=200):
    model.train()
    total_loss, total = 0.0, 0
    for step, batch in enumerate(batches.iterate(rows, batch_size, shuffle=True, rng=rng, device=device), 1):
        logits = masked_logits(model(batch["boards"], batch["castling"], batch["rating"]), batch["legal"])
        loss = F.cross_entropy(logits, batch["target"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch["target"])
        total += len(batch["target"])
        if step % log_every == 0:
            log.info("    step %d  train loss %.4f", step, total_loss / total)
    return total_loss / max(1, total)


def fit(model, batches, train_rows, val_rows, args, device, checkpoint_path, config):
    """Train with early stopping on val loss; the best model is saved to checkpoint_path."""
    rng = np.random.default_rng(args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)
    best_loss, bad_epochs = float("inf"), 0
    history = []

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_loss = train_one_epoch(model, optimizer, batches, train_rows, args.batch_size, rng, device)
        topk, val_loss = predict_topk(model, batches, val_rows, k=1, device=device)
        val_top1 = 100.0 * float((topk[:, 0] == batches.moves[val_rows]).mean())
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_top1": val_top1})
        log.info("epoch %2d  train loss %.4f  val loss %.4f  val top-1 %.1f%%  (%.0f s)",
                 epoch, train_loss, val_loss, val_top1, time.time() - started)

        if val_loss < best_loss - 1e-4:
            best_loss, bad_epochs = val_loss, 0
            torch.save({"config": config, "state_dict": model.state_dict(), "epoch": epoch,
                        "val_loss": val_loss, "history": history}, checkpoint_path)
            log.info("  saved best model to %s", checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                log.info("  no improvement for %d epochs: early stop", args.patience)
                break
    return best_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="merged .npz, e.g. dataset_all.npz")
    parser.add_argument("--model", choices=["linear", "mlp", "cnn"], default="cnn")
    parser.add_argument("--no-rating", action="store_true", help="feed a constant rating (ablation)")
    parser.add_argument("--name", default=None, help="run name (default <model>_rating / <model>_norating)")
    parser.add_argument("--hidden", type=int, default=512, help="mlp hidden units")
    parser.add_argument("--channels", type=int, default=64, help="cnn channels")
    parser.add_argument("--blocks", type=int, default=3, help="cnn residual blocks (2 convs each)")
    parser.add_argument("--epochs", type=int, default=30, help="maximum epochs")
    parser.add_argument("--patience", type=int, default=3, help="early stopping patience (epochs)")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-limit", type=int, default=0, help="use a random subset of N train rows")
    parser.add_argument("--val-limit", type=int, default=0, help="use a random subset of N val rows")
    parser.add_argument("--legal-cache", default=None, help="default: <dataset>_legal.npz")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--csv", default=None, help="default: results_<name>_val.csv")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--seed", type=int, default=0)
    add_logging_args(parser)
    args = parser.parse_args()

    name = args.name or f"{args.model}_{'norating' if args.no_rating else 'rating'}"
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint_dir) / f"{name}.pt"
    setup_logging(args.log_file or str(Path(args.checkpoint_dir) / f"{name}.log"), args.quiet)

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    data = load_dataset(args.dataset)
    flat, offsets = legal_cache.load_or_build(args.dataset, data["fens"], args.legal_cache)
    batches = MoveBatches(data, flat, offsets, use_rating=not args.no_rating)

    rng = np.random.default_rng(args.seed)
    train_rows = choose_rows(data["split"], TRAIN, args.train_limit, rng)
    val_rows = choose_rows(data["split"], VAL, args.val_limit, rng)

    config = {"name": name, "model": args.model, "use_rating": not args.no_rating, "hidden": args.hidden,
              "channels": args.channels, "blocks": args.blocks, "head_channels": 32,
              "train_rows": len(train_rows), "seed": args.seed}
    model = build_model(config).to(device)
    log.info("%s: %s parameters, %s train / %s val positions, device %s", name,
             f"{count_parameters(model):,}", f"{len(train_rows):,}", f"{len(val_rows):,}", device)

    fit(model, batches, train_rows, val_rows, args, device, checkpoint_path, config)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False)["state_dict"])
    topk, val_loss = predict_topk(model, batches, val_rows, k=3, device=device)
    log.info("best model: val loss %.4f", val_loss)
    report(result_rows(data, val_rows, topk), f"{name} on val (top-3)", args.csv or f"results_{name}_val.csv")


if __name__ == "__main__":
    main()
