"""
Board and move encoding.

This file is the single source of truth for how positions and moves are turned
into numbers. Dataset building, training and the engine (inference) must all
import from here, so the encoding can never differ between them.

Conventions
  * Everything is seen from the point of view of the player to move. If Black
    is to move, the board is mirrored (ranks flipped, colours swapped) so that
    the mover is always "White".
  * Moves are classes 0..4167 in that mirrored frame:
      - 0..4095    from_square * 64 + to_square. A queen promotion shares the
                   class of the same pawn move (queen is the default promotion).
      - 4096..4167 the 72 UNDER-promotions (knight, bishop, rook):
                   4096 + from_file * 9 + direction * 3 + piece
                   direction: 0 = capture toward the a-file, 1 = straight,
                              2 = capture toward the h-file
                   piece:     0 = knight, 1 = bishop, 2 = rook
                   (In the mirrored frame promotions always go from rank 7 to
                   rank 8, so the rank is implied. Some of the 72 classes can
                   never be legal, e.g. capturing off the board edge; the
                   legal-move mask keeps them out.)
"""
import chess
import numpy as np

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
UNDERPROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

NUM_BASE_CLASSES = 64 * 64
NUM_UNDERPROMOTION_CLASSES = 8 * 3 * 3
NUM_MOVE_CLASSES = NUM_BASE_CLASSES + NUM_UNDERPROMOTION_CLASSES  # 4168


def mirror_if_black(board):
    """Return (board_from_movers_view, was_mirrored)."""
    if board.turn == chess.BLACK:
        return board.mirror(), True
    return board, False


def encode_board(view):
    """
    Encode a board that is already in the mover's view (mover = White).

    Returns
      planes:   (12, 8, 8) uint8. Planes 0-5 = mover's pawn, knight, bishop,
                rook, queen, king; planes 6-11 = the opponent's. Indexed
                [plane, rank, file].
      castling: (4,) uint8 = [mover king-side, mover queen-side,
                opponent king-side, opponent queen-side].
    """
    planes = np.zeros((12, 8, 8), dtype=np.uint8)
    for square, piece in view.piece_map().items():
        plane = PIECE_TYPES.index(piece.piece_type)
        if piece.color != chess.WHITE:
            plane += 6
        planes[plane, chess.square_rank(square), chess.square_file(square)] = 1

    castling = np.array([
        view.has_kingside_castling_rights(chess.WHITE),
        view.has_queenside_castling_rights(chess.WHITE),
        view.has_kingside_castling_rights(chess.BLACK),
        view.has_queenside_castling_rights(chess.BLACK),
    ], dtype=np.uint8)
    return planes, castling


def encode_move(move, mirrored):
    """Move -> class index (0..4167). `mirrored` is the flag from mirror_if_black."""
    from_sq, to_sq = move.from_square, move.to_square
    if mirrored:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)

    if move.promotion in UNDERPROMOTION_PIECES:
        from_file = chess.square_file(from_sq)
        direction = chess.square_file(to_sq) - from_file + 1
        piece = UNDERPROMOTION_PIECES.index(move.promotion)
        return NUM_BASE_CLASSES + from_file * 9 + direction * 3 + piece
    return from_sq * 64 + to_sq


def decode_move(move_index, board):
    """
    Class index -> chess.Move on the REAL board (the one with the side to move,
    not the mirrored view). Used by the engine at inference time.
    """
    if move_index >= NUM_BASE_CLASSES:
        from_file, rest = divmod(move_index - NUM_BASE_CLASSES, 9)
        direction, piece = divmod(rest, 3)
        to_file = from_file + direction - 1
        if not 0 <= to_file <= 7:
            raise ValueError(f"class {move_index} is not a possible move")
        from_sq = chess.square(from_file, 6)
        to_sq = chess.square(to_file, 7)
        promotion = UNDERPROMOTION_PIECES[piece]
    else:
        from_sq, to_sq = divmod(move_index, 64)
        promotion = None

    if board.turn == chess.BLACK:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)

    if promotion is None:
        piece_on_square = board.piece_at(from_sq)
        if (piece_on_square is not None and piece_on_square.piece_type == chess.PAWN
                and chess.square_rank(to_sq) in (0, 7)):
            promotion = chess.QUEEN
    return chess.Move(from_sq, to_sq, promotion=promotion)


def legal_move_indices(view):
    """Set of class indices of all legal moves in a mirrored-view board."""
    return {encode_move(move, mirrored=False) for move in view.legal_moves}


def legal_mask(fen):
    """
    NUM_MOVE_CLASSES-long 0/1 mask of legal moves, from a stored FEN. The FEN
    stored in the dataset is already the mirrored view, so no further mirroring
    is needed.
    """
    mask = np.zeros(NUM_MOVE_CLASSES, dtype=np.uint8)
    mask[list(legal_move_indices(chess.Board(fen)))] = 1
    return mask
