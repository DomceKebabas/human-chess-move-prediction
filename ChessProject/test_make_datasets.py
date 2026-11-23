"""Tests for make_datasets.py. Run with:  python -m pytest"""
from types import SimpleNamespace

import pytest
import zstandard as zstd

from make_datasets import check_bands, generate_bands, parse_bands, sample_into_bands
from test_filters import make_game


def test_generate_bands_width_200():
    assert generate_bands(100, 700, 200) == [(100, 299), (300, 499), (500, 699), (700, 700)]


def test_generated_bands_cover_the_whole_range_without_overlap():
    bands = generate_bands(100, 3500, 200)
    assert bands[0][0] == 100 and bands[-1][1] == 3500
    assert check_bands(bands) == bands
    assert all(high - low <= 199 for low, high in bands)


def test_overlapping_bands_are_rejected():
    with pytest.raises(SystemExit):
        check_bands(parse_bands("1000-1200,1200-1400"))


def test_sample_into_bands_assigns_games_to_the_right_band(tmp_path):
    ARGS = SimpleNamespace(games_per_band=2, max_diff=100, min_plies=20, seed=0, max_scan_games=0)
    games = []
    for i in range(5):   # five games in band 1400-1599, only two will be sampled
        games += make_game(white_elo="1500", black_elo="1520", site=f"https://lichess.org/a{i:07d}")
    games += make_game(white_elo="1100", black_elo="1110", site="https://lichess.org/b0000001")
    for i in range(3):   # below the lowest band
        games += make_game(white_elo="900", black_elo="920", site=f"https://lichess.org/c{i:07d}")
    games += make_game(white_elo="1190", black_elo="1210", site="https://lichess.org/d0000001")  # players in different bands
    games += make_game(event="Rated Blitz game", site="https://lichess.org/e0000001")             # wrong time control

    path = tmp_path / "sample.pgn.zst"
    path.write_bytes(zstd.ZstdCompressor().compress("".join(games).encode("utf-8")))

    bands = [(1000, 1199), (1200, 1399), (1400, 1599)]
    names, sizes = sample_into_bands(str(path), bands, ARGS, output_dir=tmp_path)
    assert names == ["1000-1199", "1200-1399", "1400-1599"]
    assert sizes == [1, 0, 2]
    assert (tmp_path / "filtered_1000-1199.pgn").exists()
    assert not (tmp_path / "filtered_1200-1399.pgn").exists()   # empty bands write no file
    assert (tmp_path / "filtered_1400-1599.pgn").read_text().count("[Event ") == 2