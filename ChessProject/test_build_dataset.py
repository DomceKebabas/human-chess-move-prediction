"""Tests for build_dataset.py. Run with:  python -m pytest"""
import io
import random
from types import SimpleNamespace

import chess.pgn

from build_dataset import get_game_id, process_game
from encoding import legal_mask

ARGS = SimpleNamespace(min_ply=10, min_clock=0, max_per_game=1000, min_game_plies=20)

GAME_PGN = """[Event "Rated Rapid game"]
[Site "https://lichess.org/abcd1234"]
[White "alice"]
[Black "bob"]
[WhiteElo "1500"]
[BlackElo "1480"]

1. e4 { [%clk 0:10:00] } e5 { [%clk 0:10:00] } 2. Nf3 { [%clk 0:09:58] } Nc6 { [%clk 0:09:57] } 3. Bb5 { [%clk 0:09:55] } a6 { [%clk 0:09:50] } 4. Ba4 { [%clk 0:09:50] } Nf6 { [%clk 0:09:40] } 5. O-O { [%clk 0:09:45] } Be7 { [%clk 0:09:30] } 6. Re1 { [%clk 0:09:40] } b5 { [%clk 0:09:20] } 7. Bb3 { [%clk 0:09:30] } d6 { [%clk 0:09:10] } 8. c3 { [%clk 0:09:20] } O-O { [%clk 0:09:00] } 9. h3 { [%clk 0:09:10] } Nb8 { [%clk 0:08:50] } 10. d4 { [%clk 0:09:00] } Nbd7 { [%clk 0:08:40] } 11. c4 { [%clk 0:08:50] } c6 { [%clk 0:00:20] } 1-0
"""


def read(pgn_text):
    return chess.pgn.read_game(io.StringIO(pgn_text))


def test_process_game_produces_legal_examples():
    game = read(GAME_PGN)
    assert not game.errors
    examples = process_game(game, "abcd1234", ARGS, random.Random(0))
    assert len(examples) == 22 - 10          # 22 plies, first 10 skipped
    for example in examples:
        assert legal_mask(example.fen)[example.move_index] == 1
        assert example.player in ("alice", "bob")
        assert example.opponent in ("alice", "bob") and example.opponent != example.player


def test_mover_rating_and_name_are_correct():
    examples = process_game(read(GAME_PGN), "abcd1234", ARGS, random.Random(0))
    by_ply = {e.ply: e for e in examples}
    assert by_ply[10].player == "alice" and by_ply[10].rating == 1500   # ply 10 = White's move
    assert by_ply[11].player == "bob" and by_ply[11].rating == 1480


def test_low_clock_moves_are_skipped():
    args = SimpleNamespace(min_ply=10, min_clock=30, max_per_game=1000, min_game_plies=20)
    examples = process_game(read(GAME_PGN), "abcd1234", args, random.Random(0))
    assert 21 not in {e.ply for e in examples}      # Black's last move left 20 s on the clock
    assert all(e.clock >= 30 for e in examples)


def test_max_per_game_caps_examples():
    args = SimpleNamespace(min_ply=10, min_clock=0, max_per_game=3, min_game_plies=20)
    assert len(process_game(read(GAME_PGN), "abcd1234", args, random.Random(0))) == 3


def test_illegal_move_is_reported_in_game_errors():
    # python-chess does not raise on a bad move; it records it in game.errors.
    game = read('[Event "x"]\n\n1. e4 e5 2. Nf6 *\n')
    assert game.errors


def test_game_id_from_site_header():
    assert get_game_id(read(GAME_PGN)) == "abcd1234"
    assert get_game_id(read('[Event "x"]\n[Site "?"]\n\n1. e4 *\n')) == ""
    assert get_game_id(read('[Event "x"]\n\n1. e4 *\n')) == ""


def test_under_promotions_are_kept_and_legal():
    pgn = """[Event "x"]
[Site "https://lichess.org/promo001"]
[White "alice"]
[Black "bob"]
[WhiteElo "1500"]
[BlackElo "1500"]
[FEN "8/4P3/8/8/8/8/k6K/8 w - - 0 1"]
[SetUp "1"]

1. e8=N Kb3 *
"""
    game = read(pgn)
    assert not game.errors
    args = SimpleNamespace(min_ply=0, min_clock=0, max_per_game=10, min_game_plies=1)
    examples = process_game(game, "promo001", args, random.Random(0))
    first = [e for e in examples if e.ply == 0][0]
    assert first.move_index >= 4096                     # an under-promotion class
    assert legal_mask(first.fen)[first.move_index] == 1