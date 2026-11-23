"""Tests for filter_games.py. Run with:  python -m pytest"""
from types import SimpleNamespace

import zstandard as zstd

from filter_games import is_wanted_event, parse_headers, reject_reason, stream_games

ARGS = SimpleNamespace(min_rating=1000, max_rating=2000, max_diff=100, min_plies=20)


def make_game(event="Rated Rapid game", termination="Normal", white_elo="1500", black_elo="1520",
              white_title=None, num_moves=40, clock=True, last_eval="0.0", site="https://lichess.org/abc12345"):
    lines = [f'[Event "{event}"]\n', f'[Site "{site}"]\n',
             f'[WhiteElo "{white_elo}"]\n', f'[BlackElo "{black_elo}"]\n',
             '[TimeControl "600+0"]\n', f'[Termination "{termination}"]\n']
    if white_title:
        lines.append(f'[WhiteTitle "{white_title}"]\n')
    lines.append("\n")
    parts = []
    for n in range(1, num_moves + 1):
        clk = " [%clk 0:09:50]" if clock else ""
        parts.append(f"{n}. e4 {{{clk} [%eval 0.17] }} e5 {{{clk} [%eval {last_eval}] }}")
    lines.append(" ".join(parts) + " 1-0\n")
    lines.append("\n")
    return lines


def test_good_game_is_kept():
    assert reject_reason(make_game(), ARGS) is None


def test_event_matching():
    assert is_wanted_event("Rated Rapid game")
    assert is_wanted_event("Rated Classical game")
    assert is_wanted_event('[Event "Rated Rapid tournament https://lichess.org/tournament/x"]\n')
    assert not is_wanted_event("Rated Blitz game")
    assert not is_wanted_event("Rated Bullet game")
    assert not is_wanted_event("Unrated Rapid game")
    assert not is_wanted_event("Rated Correspondence game")


def test_blitz_is_rejected_and_classical_kept():
    assert reject_reason(make_game(event="Rated Blitz game"), ARGS) == "not rated rapid/classical"
    assert reject_reason(make_game(event="Rated Classical game"), ARGS) is None


def test_time_forfeit_is_kept_but_abandoned_is_not():
    assert reject_reason(make_game(termination="Time forfeit"), ARGS) is None
    assert reject_reason(make_game(termination="Abandoned"), ARGS) == "termination not allowed"


def test_bots_are_rejected():
    assert reject_reason(make_game(white_title="BOT"), ARGS) == "bot"


def test_rating_filters():
    assert reject_reason(make_game(white_elo="900"), ARGS) == "rating out of range"
    assert reject_reason(make_game(black_elo="2300"), ARGS) == "rating out of range"
    assert reject_reason(make_game(white_elo="1500", black_elo="1700"), ARGS) == "rating gap too big"
    assert reject_reason(make_game(white_elo="1500?"), ARGS) == "rating not an integer"


def test_missing_clock_data_is_rejected():
    assert reject_reason(make_game(clock=False), ARGS) == "no clock data"


def test_eval_comments_do_not_break_the_length_check():
    # Regression: "[%eval 0.0]" used to be read as move number 0, so a 41-move
    # game looked like 0 plies and was dropped as "game too short".
    assert reject_reason(make_game(num_moves=41, last_eval="0.0"), ARGS) is None
    assert reject_reason(make_game(num_moves=41, last_eval="0.17"), ARGS) is None


def test_genuinely_short_game_is_rejected():
    assert reject_reason(make_game(num_moves=5), ARGS) == "game too short"


def test_parse_headers():
    headers = parse_headers(make_game(white_elo="1234"))
    assert headers["WhiteElo"] == "1234"
    assert headers["Event"] == "Rated Rapid game"


def test_stream_games_skips_blitz_and_counts_everything(tmp_path):
    games = (make_game(event="Rated Blitz game", site="https://lichess.org/blitz001")
             + make_game(event="Rated Rapid game", site="https://lichess.org/rapid001")
             + make_game(event="Rated Classical game", site="https://lichess.org/class001"))
    path = tmp_path / "sample.pgn.zst"
    path.write_bytes(zstd.ZstdCompressor().compress("".join(games).encode("utf-8")))

    stats = {}
    yielded = list(stream_games(str(path), stats))
    assert stats["scanned"] == 3
    assert len(yielded) == 2
    assert all(is_wanted_event(game[0]) for game in yielded)