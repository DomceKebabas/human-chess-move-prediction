"""Tests for evaluation.py. Run with:  python -m pytest"""
import csv

import numpy as np

from evaluation import (by_band_table, by_player_table, hit_rows, report, sample_rows_per_band,
                        summarize, topk_from_scores, train_players)
from splits import TEST, TRAIN, VAL


def rows():
    return [
        {"band": 300, "seen_player": True, "top1": True, "topk": True},
        {"band": 300, "seen_player": False, "top1": False, "topk": True},
        {"band": 2500, "seen_player": False, "top1": False, "topk": False},
        {"band": 2500, "seen_player": True, "top1": True, "topk": True},
    ]


def test_summarize_turns_every_metric_into_a_percentage():
    [table] = summarize(rows(), lambda r: "all")
    assert table == {"group": "all", "positions": 4, "top1_pct": 50.0, "topk_pct": 75.0}


def test_by_band_table_has_labels_sorted_by_rating_and_an_overall_row():
    table = by_band_table(rows())
    assert [row["group"] for row in table] == ["300-499", "2500+", "all"]
    assert table[0]["top1_pct"] == 50.0 and table[0]["topk_pct"] == 100.0


def test_by_player_table_splits_seen_and_unseen():
    table = {row["group"]: row for row in by_player_table(rows())}
    assert table["seen player"]["top1_pct"] == 100.0
    assert table["unseen player"]["top1_pct"] == 0.0


def test_report_writes_the_band_table_to_csv(tmp_path):
    path = tmp_path / "results.csv"
    report(rows(), "test", csv_path=path)
    with open(path, newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert [row["group"] for row in written] == ["300-499", "2500+", "all"]
    assert list(written[0].keys()) == ["group", "positions", "top1_pct", "topk_pct"]


def test_train_players_includes_opponents_but_not_val_players():
    data = {
        "split": np.array([TRAIN, VAL, TEST], dtype=np.uint8),
        "players": np.array(["a", "b", "c"]),
        "opponents": np.array(["x", "y", "z"]),
    }
    assert train_players(data) == {"a", "x"}


def test_sample_rows_per_band_caps_each_band_and_is_repeatable():
    bands = [300] * 10 + [500] * 3
    rows_ = list(range(13))
    chosen = sample_rows_per_band(rows_, bands, per_band=4, seed=1)
    assert sum(bands[i] == 300 for i in chosen) == 4
    assert sum(bands[i] == 500 for i in chosen) == 3      # fewer than 4 available: keep all
    assert chosen == sample_rows_per_band(rows_, bands, per_band=4, seed=1)


def test_topk_from_scores_never_returns_illegal_moves_and_breaks_ties_by_index():
    scores = np.array([[5.0, 1.0, 1.0, 9.0]])
    legal = np.array([[True, True, True, False]])        # class 3 has the best score but is illegal
    assert topk_from_scores(scores, legal, k=3).tolist() == [[0, 1, 2]]


def test_hit_rows():
    topk = np.array([[1, 2, 3], [4, 5, 6]])
    result = hit_rows([300, 500], [True, False], [1, 6], topk)
    assert result[0] == {"band": 300, "seen_player": True, "top1": True, "topk": True}
    assert result[1] == {"band": 500, "seen_player": False, "top1": False, "topk": True}
