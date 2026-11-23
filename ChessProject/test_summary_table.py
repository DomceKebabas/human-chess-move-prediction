"""Tests for summary_table.py. Run with:  python -m pytest"""
from evaluation import write_csv
from summary_table import build_table, model_name, to_markdown


def test_model_name():
    assert model_name("results_cnn_rating_val.csv") == "cnn_rating"
    assert model_name("some/dir/results_most_common_test.csv") == "most_common"


def test_build_table_merges_files_and_variants(tmp_path):
    first = tmp_path / "results_most_common_val.csv"
    write_csv(first, [
        {"group": "300-499", "positions": 10, "all_top1_pct": 8.0, "all_topk_pct": 20.0,
         "band_top1_pct": 9.0, "band_topk_pct": 21.0},
        {"group": "all", "positions": 10, "all_top1_pct": 8.0, "all_topk_pct": 20.0,
         "band_top1_pct": 9.0, "band_topk_pct": 21.0},
    ])
    second = tmp_path / "results_cnn_rating_val.csv"
    write_csv(second, [
        {"group": "2500+", "positions": 5, "top1_pct": 50.0, "topk_pct": 80.0},
        {"group": "300-499", "positions": 5, "top1_pct": 40.0, "topk_pct": 70.0},
        {"group": "all", "positions": 10, "top1_pct": 45.0, "topk_pct": 75.0},
    ])
    groups, rows = build_table([first, second])
    assert groups == ["300-499", "2500+", "all"]
    assert [label for label, _ in rows] == ["most_common (all)", "most_common (band)", "cnn_rating"]
    markdown = to_markdown(groups, rows)
    assert "| most_common (band) | 9.0 |  | 9.0 |" in markdown
    assert "| cnn_rating | 40.0 | 50.0 | 45.0 |" in markdown
