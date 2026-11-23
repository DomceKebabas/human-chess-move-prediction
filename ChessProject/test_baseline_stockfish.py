"""Tests for baseline_stockfish.py (no real engine needed). Run with:  python -m pytest"""
import chess
import chess.engine
import numpy as np

from baseline_stockfish import analyse_rows, analyse_position, read_cache, result_rows
from encoding import encode_move

START_FEN = chess.STARTING_FEN
E2E4 = chess.E2 * 64 + chess.E4
D2D4 = chess.D2 * 64 + chess.D4
G1F3 = chess.G1 * 64 + chess.F3
A2A3 = chess.A2 * 64 + chess.A3


def info(move_uci, cp):
    return {"pv": [chess.Move.from_uci(move_uci)], "score": chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE)}


class FakeEngine:
    """Always likes e4 > d4 > Nf3; any other move restricted with root_moves scores -50."""

    def __init__(self):
        self.calls = 0

    def analyse(self, board, limit, multipv=None, root_moves=None):
        self.calls += 1
        if root_moves:
            return info(root_moves[0].uci(), -50)
        return [info("e2e4", 30), info("d2d4", 25), info("g1f3", 20)][:multipv]


def test_analyse_position_when_human_plays_the_best_move():
    engine = FakeEngine()
    record = analyse_position(engine, START_FEN, E2E4, limit=None, k=3)
    assert record == {"engine_moves": [E2E4, D2D4, G1F3], "best_cp": 30, "human_cp": 30}
    assert engine.calls == 1        # no second search needed


def test_analyse_position_evaluates_the_human_move_separately():
    record = analyse_position(FakeEngine(), START_FEN, A2A3, limit=None, k=3)
    assert record["human_cp"] == -50 and record["best_cp"] == 30


def test_engine_moves_are_in_the_dataset_class_space():
    # The stored FEN is already the mover's view, so moves are encoded unmirrored.
    assert encode_move(chess.Move.from_uci("e2e4"), mirrored=False) == E2E4


def dataset():
    return {
        "fens": np.array([START_FEN] * 3),
        "moves": np.array([E2E4, G1F3, A2A3]),
        "players": np.array(["a", "b", "c"]),
    }


def test_analyse_rows_writes_a_cache_and_resumes_from_it(tmp_path):
    cache_path = tmp_path / "cache.csv"
    engine = FakeEngine()
    analyse_rows(engine, dataset(), [0, 1], None, 3, cache_path)
    calls_after_first_run = engine.calls

    cache = analyse_rows(engine, dataset(), [0, 1, 2], None, 3, cache_path)
    assert engine.calls == calls_after_first_run + 2      # only row 2 analysed (best + human move)
    assert sorted(cache) == [0, 1, 2]
    assert read_cache(cache_path)[2]["human_cp"] == -50


def test_result_rows_top1_and_top3(tmp_path):
    cache = analyse_rows(FakeEngine(), dataset(), [0, 1, 2], None, 3, tmp_path / "cache.csv")
    rows = result_rows(dataset(), [0, 1, 2], cache, bands=[300, 300, 500], seen_players={"a"})
    assert [(r["top1"], r["topk"]) for r in rows] == [(True, True), (False, True), (False, False)]
    assert [r["seen_player"] for r in rows] == [True, False, False]
