"""
Evaluate a trained checkpoint (from train.py) on the val or test split, with
the same per-band / seen-vs-unseen tables as every baseline.

train.py already does this on val at the end of training; use this script to
re-evaluate a saved model, and in Phase 7 to run it ONCE on test.

Usage:
    python evaluate_model.py checkpoints/cnn_rating.pt dataset_all.npz
    python evaluate_model.py checkpoints/cnn_rating.pt dataset_all.npz --split test --csv results_cnn_rating_test.csv
"""
import argparse
import logging

import numpy as np
import torch
from torch.nn import functional as F

import legal_cache
from common import add_logging_args, load_dataset, setup_logging
from evaluation import bands_of, hit_rows, report, train_players
from models import build_model, masked_logits
from move_batches import MoveBatches
from splits import TEST, VAL

log = logging.getLogger(__name__)

SPLIT_CODES = {"val": VAL, "test": TEST}


def pick_device(name="auto"):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


@torch.no_grad()
def predict_topk(model, batches, rows, k=3, batch_size=1024, device="cpu"):
    """Top-k legal move classes for every row, and the mean cross-entropy loss."""
    model.eval()
    all_topk, total_loss, total = [], 0.0, 0
    for batch in batches.iterate(rows, batch_size, device=device):
        logits = masked_logits(model(batch["boards"], batch["castling"], batch["rating"]), batch["legal"])
        total_loss += F.cross_entropy(logits, batch["target"], reduction="sum").item()
        total += len(batch["target"])
        # Illegal moves are -inf, so they can only appear if fewer than k moves are legal.
        all_topk.append(torch.topk(logits, k, dim=1).indices.cpu().numpy())
    return np.concatenate(all_topk), total_loss / max(1, total)


def result_rows(data, rows, topk):
    """Result rows (band, seen_player, top1, topk) for evaluation.report."""
    bands = bands_of(data["ratings"])
    seen = train_players(data)
    return hit_rows([bands[r] for r in rows], [str(data["players"][r]) in seen for r in rows],
                    data["moves"][rows].astype(np.int64), topk)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device), checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("dataset")
    parser.add_argument("--split", choices=sorted(SPLIT_CODES), default="val",
                        help="default val; use test only for the final evaluation")
    parser.add_argument("--legal-cache", default=None, help="default: <dataset>_legal.npz")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--csv", default=None)
    add_logging_args(parser)
    args = parser.parse_args()
    setup_logging(args.log_file, args.quiet)

    device = pick_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    data = load_dataset(args.dataset)
    flat, offsets = legal_cache.load_or_build(args.dataset, data["fens"], args.legal_cache)
    batches = MoveBatches(data, flat, offsets, use_rating=config["use_rating"])

    rows = np.flatnonzero(data["split"] == SPLIT_CODES[args.split])
    topk, loss = predict_topk(model, batches, rows, args.k, device=device)
    log.info("%s loss on %s: %.4f", config["name"], args.split, loss)
    report(result_rows(data, rows, topk), f"{config['name']} on {args.split} (top-{args.k})", args.csv)


if __name__ == "__main__":
    main()
