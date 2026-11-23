"""Tests for legal_cache.py, move_batches.py, models.py and train.py. Run with:  python -m pytest"""
import random
import sys

import chess
import numpy as np
import pytest
import torch
from torch.nn import functional as F

import legal_cache
from encoding import NUM_MOVE_CLASSES, encode_board, encode_move, legal_mask, mirror_if_black
from models import build_model, masked_logits
from move_batches import MoveBatches, normalise_rating
from splits import TRAIN, VAL


def random_dataset(n, seed=0):
    """n positions from random games, encoded like build_dataset.py does it."""
    rng = random.Random(seed)
    boards, castling, moves, fens = [], [], [], []
    board = chess.Board()
    while len(fens) < n:
        if board.is_game_over() or board.ply() > 60:
            board = chess.Board()
        move = rng.choice(list(board.legal_moves))
        if board.legal_moves.count() > 1:
            view, mirrored = mirror_if_black(board)
            planes, flags = encode_board(view)
            boards.append(planes)
            castling.append(flags)
            moves.append(encode_move(move, mirrored))
            fens.append(view.fen())
        board.push(move)
    return {
        "boards": np.stack(boards), "castling": np.stack(castling),
        "moves": np.array(moves, dtype=np.int16),
        "ratings": np.array([rng.randint(400, 2800) for _ in range(n)], dtype=np.int16),
        "fens": np.array(fens),
        "players": np.array([f"p{i % 7}" for i in range(n)]),
        "opponents": np.array([f"q{i % 5}" for i in range(n)]),
        "split": np.array([TRAIN] * (n - n // 5) + [VAL] * (n // 5), dtype=np.uint8),
        "plies": np.arange(n, dtype=np.int16),
        "game_ids": np.array([f"g{i // 10}" for i in range(n)]),
    }


@pytest.fixture(scope="module")
def data():
    return random_dataset(120)


@pytest.fixture(scope="module")
def batches(data):
    flat, offsets = legal_cache.build(data["fens"])
    return MoveBatches(data, flat, offsets)


# ---- legal_cache ---------------------------------------------------------

def test_cached_masks_equal_encoding_legal_mask(data):
    flat, offsets = legal_cache.build(data["fens"])
    rows = [0, 5, 17, 119]
    masks = legal_cache.masks(flat, offsets, rows)
    for mask, row in zip(masks, rows):
        assert np.array_equal(mask, legal_mask(str(data["fens"][row])).astype(bool))


def test_cache_is_rejected_for_a_different_dataset(tmp_path, data):
    path = tmp_path / "legal.npz"
    flat, offsets = legal_cache.build(data["fens"])
    legal_cache.save(path, flat, offsets, data["fens"])
    assert legal_cache.load(path, data["fens"])[0].tolist() == flat.tolist()
    with pytest.raises(SystemExit):
        legal_cache.load(path, data["fens"][::-1])


# ---- batches ---------------------------------------------------------------

def test_batch_shapes_and_target_is_legal(batches):
    batch = batches.batch([0, 1, 2, 3])
    assert batch["boards"].shape == (4, 12, 8, 8)
    assert batch["castling"].shape == (4, 4)
    assert batch["legal"].shape == (4, NUM_MOVE_CLASSES)
    assert batch["legal"][torch.arange(4), batch["target"]].all()


def test_no_rating_feeds_a_constant(data, batches):
    without = MoveBatches(data, batches.legal_flat, batches.legal_offsets, use_rating=False)
    assert torch.equal(without.batch([0, 1])["rating"], torch.zeros(2))
    assert batches.batch([0])["rating"].item() == pytest.approx(normalise_rating([data["ratings"][0]])[0])


# ---- models ----------------------------------------------------------------

@pytest.mark.parametrize("kind", ["linear", "mlp", "cnn"])
def test_output_shape(kind, batches):
    model = build_model({"model": kind, "hidden": 32, "channels": 16, "blocks": 1})
    batch = batches.batch(list(range(8)))
    assert model(batch["boards"], batch["castling"], batch["rating"]).shape == (8, NUM_MOVE_CLASSES)


def test_masked_illegal_moves_get_probability_zero(batches):
    model = build_model({"model": "cnn", "channels": 16, "blocks": 1})
    batch = batches.batch(list(range(8)))
    probs = F.softmax(masked_logits(model(batch["boards"], batch["castling"], batch["rating"]),
                                    batch["legal"]), dim=1)
    assert torch.all(probs[~batch["legal"]] == 0)
    assert torch.allclose(probs.sum(dim=1), torch.ones(8))


def test_rating_changes_the_cnn_output(batches):
    model = build_model({"model": "cnn", "channels": 16, "blocks": 1}).eval()
    batch = batches.batch([0])
    low = model(batch["boards"], batch["castling"], torch.tensor([-1.0]))
    high = model(batch["boards"], batch["castling"], torch.tensor([1.0]))
    assert not torch.allclose(low, high)


def test_cnn_can_overfit_100_positions(data):
    """Sanity check: a model that cannot memorise 100 positions has a bug."""
    torch.manual_seed(0)
    small = random_dataset(100, seed=1)
    flat, offsets = legal_cache.build(small["fens"])
    batch = MoveBatches(small, flat, offsets).batch(np.arange(100))
    model = build_model({"model": "cnn", "channels": 32, "blocks": 2})
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(150):
        logits = masked_logits(model(batch["boards"], batch["castling"], batch["rating"]), batch["legal"])
        loss = F.cross_entropy(logits, batch["target"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = masked_logits(model(batch["boards"], batch["castling"], batch["rating"]), batch["legal"])
    accuracy = (logits.argmax(dim=1) == batch["target"]).float().mean().item()
    assert accuracy > 0.95


# ---- train.py end to end ---------------------------------------------------

def test_train_script_end_to_end(tmp_path, monkeypatch, data):
    import evaluate_model
    import train

    dataset = tmp_path / "tiny.npz"
    np.savez_compressed(dataset, **data)
    results = tmp_path / "results.csv"
    monkeypatch.setattr(sys, "argv", ["train.py", str(dataset), "--model", "linear", "--epochs", "2",
                                      "--batch-size", "32", "--checkpoint-dir", str(tmp_path / "ckpt"),
                                      "--csv", str(results), "--quiet"])
    train.main()
    checkpoint = tmp_path / "ckpt" / "linear_rating.pt"
    assert checkpoint.exists() and results.exists()
    assert (tmp_path / "tiny_legal.npz").exists()        # legal cache built next to the dataset

    model, saved = evaluate_model.load_checkpoint(checkpoint)
    assert saved["config"]["use_rating"] is True
    assert saved["config"]["model"] == "linear"
