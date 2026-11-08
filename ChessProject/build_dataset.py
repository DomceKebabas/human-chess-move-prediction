"""
Step 2: turn filtered.pgn into training examples.

Each example = (board before the move, rating of the player who moved, move played).

Position-level filters:
  - skip the first --min-ply plies (opening theory)
  - skip moves played with less than --min-clock seconds left
  - skip positions with only one legal move
  - skip under-promotions (rare; keeps move encoding simple)
  - keep at most --max-per-game random positions per game

Encoding (always from the point of view of the player to move):
  - If Black is to move, the board is mirrored so the mover is always "White".
    The model then only has to learn one orientation.
  - boards:  (N, 12, 8, 8) uint8. Planes 0-5 = mover's P,N,B,R,Q,K; 6-11 = opponent's.
  - extras:  (N, 4) uint8 = castling rights [mover K-side, mover Q-side, opp K-side, opp Q-side]
  - moves:   (N,) int16 = from_square * 64 + to_square  (0..4095), in the mirrored frame
  - fens:    mirrored position as FEN string, used later to rebuild the legal-move mask
  - split:   0 = train, 1 = val, 2 = test, decided per GAME (hash of the game id)

Usage:
  python build_dataset.py filtered.pgn dataset.npz --max-games 50000
"""
import argparse
import random
import zlib

import chess
import chess.pgn
import numpy as np

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


# ---------- encoding helpers (reuse these in your training code) ----------

def mirror_if_black(board):
    """Return (board_from_movers_view, was_mirrored)."""
    if board.turn == chess.BLACK:
        return board.mirror(), True
    return board, False


def encode_board(board):
    """board must already be from the mover's view (mover = White)."""
    planes = np.zeros((12, 8, 8), dtype=np.uint8)
    for sq, piece in board.piece_map().items():
        idx = PIECE_TYPES.index(piece.piece_type) + (0 if piece.color == chess.WHITE else 6)
        planes[idx, sq // 8, sq % 8] = 1
    extras = np.array([
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    ], dtype=np.uint8)
    return planes, extras


def encode_move(move, mirrored):
    f, t = move.from_square, move.to_square
    if mirrored:
        f ^= 56  # vertical flip of a square index
        t ^= 56
    return f * 64 + t


def legal_mask(fen):
    """4096-long 0/1 mask of legal moves for a stored (mirrored) FEN."""
    board = chess.Board(fen)
    mask = np.zeros(4096, dtype=np.uint8)
    for m in board.legal_moves:
        mask[m.from_square * 64 + m.to_square] = 1
    return mask


def split_of(game_id):
    """Deterministic 80/10/10 split by game id (same game -> same split, always)."""
    h = zlib.crc32(game_id.encode()) % 10
    return 0 if h < 8 else (1 if h == 8 else 2)


# ---------- game processing ----------

def process_game(game, args, rng):
    h = game.headers
    try:
        elo = {chess.WHITE: int(h["WhiteElo"]), chess.BLACK: int(h["BlackElo"])}
    except (KeyError, ValueError):
        return []

    nodes = list(game.mainline())
    if len(nodes) < args.min_game_plies:
        return []

    game_id = h.get("Site", "").rsplit("/", 1)[-1] or str(id(game))
    board = game.board()
    candidates = []  # (ply, fen_before_move, move, clock_after_move, mover_color)

    for ply, node in enumerate(nodes):
        move = node.move
        if ply >= args.min_ply:
            clock = node.clock()  # seconds left AFTER this move (from [%clk]), or None
            clock_ok = (clock is not None and clock >= args.min_clock) or args.min_clock <= 0
            promo_ok = move.promotion in (None, chess.QUEEN)
            if clock_ok and promo_ok and board.legal_moves.count() > 1:
                candidates.append((ply, board.fen(), move, -1.0 if clock is None else clock, board.turn))
        board.push(move)

    if len(candidates) > args.max_per_game:
        candidates = rng.sample(candidates, args.max_per_game)

    rows = []
    for ply, fen, move, clock, color in candidates:
        board_before = chess.Board(fen)
        view, mirrored = mirror_if_black(board_before)
        # sanity check: the played move must be legal in the (mirrored) frame
        move_idx = encode_move(move, mirrored)
        assert legal_mask(view.fen())[move_idx] == 1, f"encoding bug in game {game_id}, ply {ply}"
        planes, extras = encode_board(view)
        rows.append((planes, extras, move_idx, elo[color], clock, ply, game_id, view.fen()))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="filtered .pgn from step 1")
    p.add_argument("output", help="output .npz file")
    p.add_argument("--min-ply", type=int, default=10)
    p.add_argument("--min-clock", type=float, default=30.0,
                   help="skip moves with fewer seconds left; 0 keeps everything")
    p.add_argument("--max-per-game", type=int, default=15)
    p.add_argument("--min-game-plies", type=int, default=20)
    p.add_argument("--max-games", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    all_rows = []
    games_read = 0

    with open(args.input, encoding="utf-8", errors="replace") as f:
        while games_read < args.max_games:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games_read += 1
            all_rows.extend(process_game(game, args, rng))
            if games_read % 1000 == 0:
                print(f"games read: {games_read:,}   positions: {len(all_rows):,}")

    if not all_rows:
        raise SystemExit("No positions kept. Check your filters and the input file.")

    boards = np.stack([r[0] for r in all_rows])
    extras = np.stack([r[1] for r in all_rows])
    moves = np.array([r[2] for r in all_rows], dtype=np.int16)
    ratings = np.array([r[3] for r in all_rows], dtype=np.int16)
    clocks = np.array([r[4] for r in all_rows], dtype=np.float32)
    plies = np.array([r[5] for r in all_rows], dtype=np.int16)
    game_ids = np.array([r[6] for r in all_rows])
    fens = np.array([r[7] for r in all_rows])
    split = np.array([split_of(g) for g in game_ids], dtype=np.uint8)

    np.savez_compressed(args.output, boards=boards, extras=extras, moves=moves,
                        ratings=ratings, clocks=clocks, plies=plies,
                        game_ids=game_ids, fens=fens, split=split)

    print(f"\nDone. {games_read:,} games -> {len(moves):,} positions")
    print(f"train/val/test positions: {(split == 0).sum():,} / {(split == 1).sum():,} / {(split == 2).sum():,}")
    print(f"mean rating: {ratings.mean():.0f}, rating range: {ratings.min()}-{ratings.max()}")
    # leakage check: no game may appear in two splits
    g_split = {}
    for g, s in zip(game_ids, split):
        assert g_split.setdefault(g, s) == s, "game found in two splits!"
    print("Leakage check passed: every game is in exactly one split.")


if __name__ == "__main__":
    main()