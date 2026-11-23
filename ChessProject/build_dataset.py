"""
build_dataset.py - turn a filtered .pgn into training examples.

Each example = (board before the move, rating of the player who moved, move played).

Position-level filters
  * skip the first --min-ply plies (opening theory)
  * skip moves played with less than --min-clock seconds left (0 keeps all)
  * skip positions with only one legal move
  * keep at most --max-per-game random positions per game

Game-level safety
  * games that python-chess could only half-parse are skipped
  * games without a Lichess id, and repeated game ids, are skipped

Encoding lives in encoding.py; the split and overlap report in splits.py.

Saved arrays (.npz)
  boards     (N, 12, 8, 8) uint8   mover's view, see encoding.encode_board
  castling   (N, 4) uint8
  moves      (N,) int16            move class 0..4167, see encoding.py
  ratings    (N,) int16            rating of the player who moved
  clocks     (N,) float32          seconds left after the move (-1 if unknown)
  plies      (N,) int16
  game_ids   (N,) str
  fens       (N,) str              mirrored position, used to rebuild legal masks
  players    (N,) str              name of the player who moved
  opponents  (N,) str
  split      (N,) uint8            0 train, 1 val, 2 test (per game)

Usage:
  python build_dataset.py filtered.pgn dataset.npz --max-games 50000
"""
import argparse
import logging
import random
from collections import Counter
from typing import NamedTuple

import chess
import chess.pgn
import numpy as np

from common import add_logging_args, setup_logging
from encoding import encode_board, encode_move, legal_move_indices, mirror_if_black
from splits import overlap_report, split_of

log = logging.getLogger(__name__)


class Example(NamedTuple):
    planes: np.ndarray
    castling: np.ndarray
    move_index: int
    rating: int
    clock: float
    ply: int
    game_id: str
    fen: str
    player: str
    opponent: str


def get_game_id(game):
    """Lichess game id from the Site header, or '' if there is none."""
    site = game.headers.get("Site", "")
    game_id = site.rsplit("/", 1)[-1]
    return "" if game_id == "?" else game_id


def process_game(game, game_id, args, rng):
    headers = game.headers
    try:
        ratings = {chess.WHITE: int(headers["WhiteElo"]), chess.BLACK: int(headers["BlackElo"])}
    except (KeyError, ValueError):
        return []
    names = {chess.WHITE: headers.get("White", "?"), chess.BLACK: headers.get("Black", "?")}

    nodes = list(game.mainline())
    if len(nodes) < args.min_game_plies:
        return []

    board = game.board()
    candidates = []  # (ply, fen_before_move, move, clock_after_move, mover_color)
    for ply, node in enumerate(nodes):
        move = node.move
        if ply >= args.min_ply:
            clock = node.clock()  # seconds left AFTER the move, from [%clk]; None if absent
            clock_ok = args.min_clock <= 0 or (clock is not None and clock >= args.min_clock)
            if clock_ok and board.legal_moves.count() > 1:
                candidates.append((ply, board.fen(), move, -1.0 if clock is None else clock, board.turn))
        board.push(move)

    if len(candidates) > args.max_per_game:
        candidates = rng.sample(candidates, args.max_per_game)

    examples = []
    for ply, fen, move, clock, mover_color in candidates:
        view, mirrored = mirror_if_black(chess.Board(fen))
        move_index = encode_move(move, mirrored)
        if move_index not in legal_move_indices(view):
            raise ValueError(f"encoding bug in game {game_id}, ply {ply}")
        planes, castling = encode_board(view)
        examples.append(Example(planes, castling, move_index, ratings[mover_color], clock, ply,
                                game_id, view.fen(), names[mover_color], names[not mover_color]))
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="filtered .pgn")
    parser.add_argument("output", help="output .npz file")
    parser.add_argument("--min-ply", type=int, default=10)
    parser.add_argument("--min-clock", type=float, default=30.0,
                        help="skip moves with fewer seconds left; 0 keeps everything")
    parser.add_argument("--max-per-game", type=int, default=15)
    parser.add_argument("--min-game-plies", type=int, default=20)
    parser.add_argument("--max-games", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    add_logging_args(parser)
    args = parser.parse_args()
    setup_logging(args.log_file, args.quiet)

    rng = random.Random(args.seed)
    examples = []
    seen_game_ids = set()
    skipped = Counter()
    games_read = 0

    with open(args.input, encoding="utf-8", errors="replace") as pgn_file:
        while games_read < args.max_games:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            games_read += 1

            if game.errors:
                skipped["half-parsed game (bad move)"] += 1
                continue
            game_id = get_game_id(game)
            if not game_id:
                skipped["no game id"] += 1
                continue
            if game_id in seen_game_ids:
                skipped["repeated game id"] += 1
                continue
            seen_game_ids.add(game_id)

            examples.extend(process_game(game, game_id, args, rng))
            if games_read % 1000 == 0:
                log.info("games read: %s   positions: %s", f"{games_read:,}", f"{len(examples):,}")

    if not examples:
        raise SystemExit("No positions kept. Check your filters and the input file.")

    arrays = {
        "boards": np.stack([e.planes for e in examples]),
        "castling": np.stack([e.castling for e in examples]),
        "moves": np.array([e.move_index for e in examples], dtype=np.int16),
        "ratings": np.array([e.rating for e in examples], dtype=np.int16),
        "clocks": np.array([e.clock for e in examples], dtype=np.float32),
        "plies": np.array([e.ply for e in examples], dtype=np.int16),
        "game_ids": np.array([e.game_id for e in examples]),
        "fens": np.array([e.fen for e in examples]),
        "players": np.array([e.player for e in examples]),
        "opponents": np.array([e.opponent for e in examples]),
        "split": np.array([split_of(e.game_id) for e in examples], dtype=np.uint8),
    }
    np.savez_compressed(args.output, **arrays)

    split = arrays["split"]
    ratings = arrays["ratings"]
    log.info("Done. %s games -> %s positions", f"{games_read:,}", f"{len(examples):,}")
    for reason, count in skipped.items():
        log.info("  skipped games (%s): %s", reason, f"{count:,}")
    log.info("train/val/test positions: %s / %s / %s",
             f"{int((split == 0).sum()):,}", f"{int((split == 1).sum()):,}", f"{int((split == 2).sum()):,}")
    log.info("mean rating: %.0f, rating range: %d-%d", ratings.mean(), ratings.min(), ratings.max())
    overlap_report(arrays)


if __name__ == "__main__":
    main()