"""Tests for encoding.py and splits.py. Run with:  python -m pytest"""
import chess
import numpy as np

from encoding import (NUM_BASE_CLASSES, NUM_MOVE_CLASSES, decode_move, encode_board,
                      encode_move, legal_mask, legal_move_indices, mirror_if_black)
from splits import TEST, TRAIN, VAL, overlap_report, position_key, split_of


# ---------- mirroring ----------

def test_white_to_move_is_not_mirrored():
    board = chess.Board()
    view, mirrored = mirror_if_black(board)
    assert not mirrored
    assert view.fen() == board.fen()


def test_black_to_move_is_mirrored_so_mover_is_white():
    board = chess.Board()
    board.push_san("e4")
    view, mirrored = mirror_if_black(board)
    assert mirrored and view.turn == chess.WHITE
    assert view.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE)  # Black's king, seen as White's
    assert view.piece_at(chess.E5) == chess.Piece(chess.PAWN, chess.BLACK)  # White's e4 pawn, seen as Black's


# ---------- encode_move ----------

def test_encode_move_white():
    assert encode_move(chess.Move.from_uci("e2e4"), mirrored=False) == chess.E2 * 64 + chess.E4


def test_encode_move_black_is_mirrored():
    # Black's e7-e5 looks like e2-e4 from the mover's view.
    assert encode_move(chess.Move.from_uci("e7e5"), mirrored=True) == chess.E2 * 64 + chess.E4


def _assert_encoded_move_is_legal(fen, uci):
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves
    view, mirrored = mirror_if_black(board)
    assert encode_move(move, mirrored) in legal_move_indices(view)
    return board, move


def test_castling_white_and_black():
    _assert_encoded_move_is_legal("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", "e1g1")
    _assert_encoded_move_is_legal("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", "e1c1")
    _assert_encoded_move_is_legal("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1", "e8g8")
    _assert_encoded_move_is_legal("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1", "e8c8")


def test_en_passant():
    fen = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
    _assert_encoded_move_is_legal(fen, "e5f6")


def test_en_passant_black_to_move():
    fen = "rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3"
    _assert_encoded_move_is_legal(fen, "e4d3")


def test_queen_promotion_white_and_black():
    for fen, uci in (("8/4P3/8/8/8/8/k6K/8 w - - 0 1", "e7e8q"),
                     ("8/8/8/8/8/K6k/4p3/8 b - - 0 1", "e2e1q")):
        board, move = _assert_encoded_move_is_legal(fen, uci)
        view, mirrored = mirror_if_black(board)
        assert decode_move(encode_move(move, mirrored), board) == move


def test_queen_promotion_uses_the_plain_pawn_move_class():
    index = encode_move(chess.Move.from_uci("e7e8q"), mirrored=False)
    assert index == chess.E7 * 64 + chess.E8 and index < NUM_BASE_CLASSES


def test_every_promotion_piece_has_its_own_class():
    indices = {encode_move(chess.Move.from_uci("e7e8" + piece), False) for piece in "qrbn"}
    assert len(indices) == 4
    assert sum(index >= NUM_BASE_CLASSES for index in indices) == 3   # r, b, n are the extra classes


PROMOTION_POSITIONS = [
    # (fen, uci moves that must all be legal and round-trip exactly)
    ("3r1r2/4P3/8/8/8/8/k6K/8 w - - 0 1",
     ["e7e8q", "e7e8r", "e7e8b", "e7e8n", "e7d8q", "e7d8r", "e7d8b", "e7d8n",
      "e7f8q", "e7f8r", "e7f8b", "e7f8n"]),
    ("8/8/8/8/8/K6k/4p3/3R1R2 b - - 0 1",
     ["e2e1q", "e2e1r", "e2e1b", "e2e1n", "e2d1q", "e2d1r", "e2d1b", "e2d1n",
      "e2f1q", "e2f1r", "e2f1b", "e2f1n"]),
    ("8/P7/8/8/8/8/k6K/1r6 w - - 0 1", ["a7a8q", "a7a8n", "a7a8r", "a7a8b"]),     # edge file
    ("8/8/8/8/8/K6k/7p/6R1 b - - 0 1", ["h2h1q", "h2h1n", "h2g1q", "h2g1b"]),      # edge file, capture
]


def test_every_promotion_round_trips_for_both_colours():
    for fen, ucis in PROMOTION_POSITIONS:
        board = chess.Board(fen)
        view, mirrored = mirror_if_black(board)
        for uci in ucis:
            move = chess.Move.from_uci(uci)
            assert move in board.legal_moves, (fen, uci)
            index = encode_move(move, mirrored)
            assert 0 <= index < NUM_MOVE_CLASSES
            assert index in legal_move_indices(view)
            assert decode_move(index, board) == move, (fen, uci)


def test_all_legal_promotions_get_distinct_classes():
    for fen, _ in PROMOTION_POSITIONS:
        board = chess.Board(fen)
        view, mirrored = mirror_if_black(board)
        legal = list(board.legal_moves)
        indices = [encode_move(move, mirrored) for move in legal]
        # Queen promotion and non-promotion never collide inside one position,
        # so every legal move must map to its own class.
        assert len(set(indices)) == len(legal), fen


def test_impossible_under_promotion_class_cannot_be_decoded():
    # class: from_file 0, direction 0 (capture toward the a-file), knight -> off the board
    impossible = NUM_BASE_CLASSES + 0 * 9 + 0 * 3 + 0
    try:
        decode_move(impossible, chess.Board())
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# ---------- decode_move round trip on a real game (both colours, castling) ----------

def test_round_trip_over_a_whole_game():
    san_moves = ("e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O "
                 "h3 Nb8 d4 Nbd7 c4 c6 Nbd2 Bb7 Bc2 Re8").split()
    board = chess.Board()
    for san in san_moves:
        move = board.parse_san(san)
        view, mirrored = mirror_if_black(board)
        index = encode_move(move, mirrored)
        assert index in legal_move_indices(view)
        assert decode_move(index, board) == move
        board.push(move)


# ---------- encode_board and legal_mask ----------

def test_start_position_planes():
    planes, castling = encode_board(chess.Board())
    assert planes.shape == (12, 8, 8)
    assert planes[0, 1, :].sum() == 8          # mover's pawns on rank 2
    assert planes[6, 6, :].sum() == 8          # opponent's pawns on rank 7
    assert planes[5, 0, 4] == 1                # mover's king on e1
    assert planes.sum() == 32
    assert castling.tolist() == [1, 1, 1, 1]


def test_castling_rights_are_from_movers_view():
    board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b Kq - 0 1")  # Black to move
    view, _ = mirror_if_black(board)
    _, castling = encode_board(view)
    assert castling.tolist() == [0, 1, 1, 0]  # mover (Black) has only queen-side; opponent only king-side


def test_legal_mask_start_position():
    mask = legal_mask(chess.Board().fen())
    assert mask.shape == (NUM_MOVE_CLASSES,) == (4168,)
    assert mask.sum() == 20


# ---------- splits ----------

def test_split_is_deterministic_and_roughly_80_10_10():
    ids = [f"game{i}" for i in range(10_000)]
    assert [split_of(i) for i in ids] == [split_of(i) for i in ids]
    counts = np.bincount([split_of(i) for i in ids], minlength=3) / len(ids)
    assert abs(counts[TRAIN] - 0.8) < 0.03
    assert abs(counts[VAL] - 0.1) < 0.02
    assert abs(counts[TEST] - 0.1) < 0.02


def test_position_key_ignores_move_counters():
    a = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    b = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 7 22"
    assert position_key(a) == position_key(b)


def test_overlap_report_counts_shared_positions_and_players():
    fen_shared = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    fen_other = "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1"
    data = {
        "split": np.array([TRAIN, TRAIN, TEST, TEST], dtype=np.uint8),
        "fens": np.array([fen_shared, fen_other, fen_shared + "", fen_other.replace("3P4", "2P5")]),
        "game_ids": np.array(["a", "b", "c", "d"]),
        "plies": np.array([10, 10, 10, 10]),
        "players": np.array(["alice", "bob", "alice", "carol"]),
        "opponents": np.array(["bob", "alice", "dave", "erin"]),
    }
    report = overlap_report(data)
    assert report["duplicate_rows"] == 0
    assert report["test_positions_also_in_train_pct"] == 50.0
    assert report["test_movers_also_in_train_games_pct"] == 50.0  # alice yes, carol no