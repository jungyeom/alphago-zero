"""
Python wrapper around the C++ Board that matches the Python Board interface.

This lets all existing code (Game, MCTS, self-play, server) use the C++ board
without any changes — just swap `Board` for `CppBoard`.

Falls back to the Python Board if the C++ module isn't available.
"""

import numpy as np

try:
    import sys
    import os
    # Add project root to path so alphago_core.so can be found
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import alphago_core as _ac
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

from go_engine.board import EMPTY, BLACK, WHITE, OPPONENT


def _to_move(move):
    """Convert Python move (tuple or None) to C++ Move."""
    if move is None:
        return _ac.pass_move()
    return _ac.board_move(move[0], move[1])


def _from_move(cmove):
    """Convert C++ Move to Python move (tuple or None)."""
    if cmove.is_pass():
        return None
    return (cmove.row, cmove.col)


class CppBoard:
    """
    Drop-in replacement for go_engine.board.Board using C++ backend.

    Exposes the same interface:
      - grid: numpy array (size, size) with 0=empty, 1=black, 2=white
      - play(color, move): place a stone or pass
      - is_legal(color, move): check legality
      - legal_moves(color): list of legal moves including pass (None)
      - score(): dict with scoring info
      - copy(): deep copy
      - captured: dict {BLACK: n, WHITE: n}
      - komi: float
      - move_history: list of (color, move) tuples
      - size: int (property)
    """

    def __init__(self, size: int = 13):
        if not HAS_CPP:
            raise RuntimeError("C++ module not available. Run ./build_cpp.sh first.")
        self._board = _ac.CBoard(size)
        self._size = size
        self.move_history: list[tuple[int, tuple[int, int] | None]] = []
        self.captured = {BLACK: 0, WHITE: 0}
        self.komi = 6.5
        self._board.komi = 6.5

    @property
    def size(self) -> int:
        return self._size

    @property
    def grid(self) -> np.ndarray:
        return self._board.grid

    def copy(self) -> "CppBoard":
        new = CppBoard.__new__(CppBoard)
        new._board = self._board.copy()
        new._size = self._size
        new.move_history = self.move_history.copy()
        new.captured = self.captured.copy()
        new.komi = self.komi
        return new

    def is_legal(self, color: int, move: tuple[int, int] | None) -> bool:
        return self._board.is_legal(color, _to_move(move))

    def legal_moves(self, color: int) -> list[tuple[int, int] | None]:
        cmoves = self._board.legal_moves(color)
        return [_from_move(m) for m in cmoves]

    def play(self, color: int, move: tuple[int, int] | None) -> None:
        if not self.is_legal(color, move):
            raise ValueError(
                f"Illegal move: {move} for {'black' if color == BLACK else 'white'}"
            )
        self._board.play(color, _to_move(move))
        self.move_history.append((color, move))
        # Sync capture counts from C++ board
        self.captured[BLACK] = self._board.captured_by(BLACK)
        self.captured[WHITE] = self._board.captured_by(WHITE)

    def score(self) -> dict:
        s = self._board.score()
        # Ensure komi is applied (C++ board has its own komi)
        return {
            "black": s["black"],
            "white": s["white"],
            "winner": s["winner"],
            "margin": s["margin"],
        }

    def _group(self, row: int, col: int):
        """Compatibility: returns (stones, liberties) sets."""
        # Use the Python implementation for this rarely-called method
        from go_engine.board import Board as PyBoard
        py = PyBoard(size=self._size)
        py.grid = self.grid.copy()
        return py._group(row, col)

    def _liberties(self, row: int, col: int) -> int:
        _, libs = self._group(row, col)
        return len(libs)

    def _toggle_hash(self, row: int, col: int, color: int) -> None:
        pass  # Not needed externally with C++ board

    def __str__(self) -> str:
        symbols = {EMPTY: ".", BLACK: "X", WHITE: "O"}
        col_labels = "  " + " ".join(
            chr(ord("A") + i + (1 if i >= 8 else 0)) for i in range(self._size)
        )
        rows = [col_labels]
        grid = self.grid
        for r in range(self._size):
            row_label = f"{self._size - r:2d}"
            row_str = " ".join(symbols[grid[r, c]] for c in range(self._size))
            rows.append(f"{row_label} {row_str} {row_label}")
        rows.append(col_labels)
        return "\n".join(rows)


def get_board_class():
    """Return CppBoard if C++ is available, otherwise Python Board."""
    if HAS_CPP:
        return CppBoard
    from go_engine.board import Board
    return Board
