import chess
import numpy as np


def board_to_array(board):
    pieces = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
              'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
    array = np.zeros((8, 8, 12), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            array[row, col, pieces[piece.symbol()]] = 1

    return array


def move_to_index(move: chess.Move) -> int:
    from_sq = move.from_square
    to_sq = move.to_square

    if move.promotion:
        promo_offset = {
            chess.QUEEN: 0,
            chess.ROOK: 1,
            chess.BISHOP: 2,
            chess.KNIGHT: 3
        }[move.promotion]
        idx = 1792 + (from_sq - chess.A7) * 16 + (to_sq - chess.A8) * 4 + promo_offset
    else:
        idx = from_sq * 64 + to_sq

    return idx % 1968  # Garantisce che sia sempre nel range