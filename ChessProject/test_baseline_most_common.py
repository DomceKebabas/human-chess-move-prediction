"""Tests for baseline_most_common.py. Run with:  python -m pytest"""
from collections import Counter

import chess
import numpy as np

from bands import band_label, band_of
from baseline_most_common import evaluate, rank_moves
from encoding import legal_move_indices
from splits import TRAIN, VAL

START_FEN = chess.STARTING_FEN          # White to move, so it is already the mover's view
E2E4 = chess.E2 * 64 + chess.E4         # move classes (from * 64 + to)
D2D4 = chess.D2 * 64 + chess.D4
G1F3 = chess.G1 * 64 + chess.F3


def test_band_of():
    assert band_of(300) == 300
    assert band_of(499) == 300
    assert band_of(500) == 500
    assert band_of(1234) == 1100
    assert band_of(2699) == 2500
    assert band_of(2750) == 2500        # 2700-2899 is merged into the top band


def test_band_label():
    assert band_label(300) == "300-499"
    assert band_label(2500) == "2500+"


def test_rank_moves_uses_counts_and_breaks_ties_by_class():
    counts = Counter({D2D4: 5, E2E4: 5, G1F3: 1})
    # d2d4 and e2e4 are tied; d2d4 has the smaller class index, so it comes first.
    assert rank_moves(START_FEN, [counts], Counter(), k=3) == sorted([D2D4, E2E4]) + [G1F3]


def test_rank_moves_fills_unseen_position_with_legal_moves_only():
    # The position was never seen. The most common move overall is a class that
    # is NOT legal here (a king move e1e2), so it must be skipped.
    illegal = chess.E1 * 64 + chess.E2
    overall = Counter({illegal: 100, G1F3: 10, E2E4: 5})
    guesses = rank_moves(START_FEN, [None], overall, k=3)
    assert guesses[:2] == [G1F3, E2E4]
    assert len(guesses) == 3
    assert set(guesses) <= legal_move_indices(chess.Board(START_FEN))


def test_specific_counter_comes_before_general_one():
    band_counts = Counter({E2E4: 1})
    all_counts = Counter({D2D4: 50})
    assert rank_moves(START_FEN, [band_counts, all_counts], Counter(), k=2) == [E2E4, D2D4]


def test_by_band_variant_uses_the_players_own_band():
    # Train: weak players (band 300) play e4 twice, strong players (band 1500)
    # play d4 three times. Validation: one weak player in the same position.
    # all-ratings should guess d4 (wrong); by-band should guess e4 (right).
    data = {
        "split": np.array([TRAIN, TRAIN, TRAIN, TRAIN, TRAIN, VAL], dtype=np.uint8),
        "fens": np.array([START_FEN] * 6),
        "moves": np.array([E2E4, E2E4, D2D4, D2D4, D2D4, E2E4]),
        "ratings": np.array([350, 360, 1550, 1560, 1570, 340]),
        "players": np.array(["a", "b", "c", "d", "e", "a"]),
        "opponents": np.array(["x", "x", "y", "y", "y", "z"]),
    }
    [row] = evaluate(data, VAL, k=3)
    assert row["band"] == 300
    assert row["seen_position"] and row["seen_player"]
    assert not row["all_top1"] and row["all_topk"]
    assert row["band_top1"]
