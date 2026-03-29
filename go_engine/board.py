"""
Go board engine for 13x13 (configurable size).

Board representation:
  - 2D numpy array: 0=empty, 1=black, 2=white
  - Zobrist hashing for fast position comparison and ko detection

Coordinate system:
  - (row, col) where (0,0) is top-left
  - "pass" is represented as None
"""

import numpy as np
from collections import deque

EMPTY = 0
BLACK = 1
WHITE = 2

OPPONENT = {BLACK: WHITE, WHITE: BLACK}


def _init_zobrist(size: int) -> np.ndarray:
    """Create Zobrist hash table: random 64-bit ints for each (row, col, color)."""
    rng = np.random.RandomState(42)  # fixed seed for reproducibility
    # shape: (size, size, 3) — index 0 unused, 1=black, 2=white
    return rng.randint(0, 2**63, size=(size, size, 3), dtype=np.uint64)


class Board:
    """
    Go board with full rules: captures, ko, legal move generation, scoring.

    Usage:
        board = Board(size=13)
        board.play(BLACK, (3, 4))
        board.play(WHITE, (2, 3))
        board.play(BLACK, None)  # pass
        legal = board.legal_moves(BLACK)
        score = board.score()
    """

    def __init__(self, size: int = 13):
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.int8)
        self._zobrist_table = _init_zobrist(size)
        self._hash = np.uint64(0)
        # Store full position history for superko detection
        self._position_history: set[int] = set()
        self._position_history.add(int(self._hash))
        self.move_history: list[tuple[int, tuple[int, int] | None]] = []
        self.captured = {BLACK: 0, WHITE: 0}  # stones captured BY each color
        self.komi = 6.5  # compensation for white

    def copy(self) -> "Board":
        """Return a deep copy of this board."""
        new = Board.__new__(Board)
        new.size = self.size
        new.grid = self.grid.copy()
        new._zobrist_table = self._zobrist_table  # shared, immutable
        new._hash = self._hash
        new._position_history = self._position_history.copy()
        new.move_history = self.move_history.copy()
        new.captured = self.captured.copy()
        new.komi = self.komi
        return new

    # ── Neighbors & Groups ──────────────────────────────────────────

    def _neighbors(self, row: int, col: int) -> list[tuple[int, int]]:
        """Return orthogonal neighbors within bounds."""
        result = []
        if row > 0:
            result.append((row - 1, col))
        if row < self.size - 1:
            result.append((row + 1, col))
        if col > 0:
            result.append((row, col - 1))
        if col < self.size - 1:
            result.append((row, col + 1))
        return result

    def _group(self, row: int, col: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        """
        BFS to find the connected group containing (row, col) and its liberties.

        Returns:
            (stones, liberties) — sets of coordinates
        """
        color = self.grid[row, col]
        if color == EMPTY:
            return set(), set()

        stones = set()
        liberties = set()
        queue = deque([(row, col)])
        stones.add((row, col))

        while queue:
            r, c = queue.popleft()
            for nr, nc in self._neighbors(r, c):
                if (nr, nc) in stones:
                    continue
                cell = self.grid[nr, nc]
                if cell == EMPTY:
                    liberties.add((nr, nc))
                elif cell == color:
                    stones.add((nr, nc))
                    queue.append((nr, nc))

        return stones, liberties

    def _liberties(self, row: int, col: int) -> int:
        """Count liberties of the group at (row, col)."""
        _, libs = self._group(row, col)
        return len(libs)

    # ── Zobrist Hashing ─────────────────────────────────────────────

    def _toggle_hash(self, row: int, col: int, color: int) -> None:
        """XOR the zobrist value for placing/removing a stone."""
        self._hash ^= self._zobrist_table[row, col, color]

    # ── Stone Placement & Capture ───────────────────────────────────

    def _place_stone(self, color: int, row: int, col: int) -> int:
        """
        Place a stone and capture any opponent groups with zero liberties.

        Returns:
            Number of stones captured.
        """
        self.grid[row, col] = color
        self._toggle_hash(row, col, color)

        captured_count = 0
        opp = OPPONENT[color]

        # Check opponent neighbors for captures
        for nr, nc in self._neighbors(row, col):
            if self.grid[nr, nc] == opp:
                stones, liberties = self._group(nr, nc)
                if len(liberties) == 0:
                    captured_count += len(stones)
                    for sr, sc in stones:
                        self.grid[sr, sc] = EMPTY
                        self._toggle_hash(sr, sc, opp)

        return captured_count

    def _is_suicide(self, color: int, row: int, col: int) -> bool:
        """
        Check if placing color at (row, col) would be suicide.
        A move is suicide if it results in the placed stone's group having zero
        liberties AND it doesn't capture any opponent stones.
        """
        # Temporarily place the stone
        self.grid[row, col] = color

        # Check if any opponent neighbor would be captured
        opp = OPPONENT[color]
        captures = False
        for nr, nc in self._neighbors(row, col):
            if self.grid[nr, nc] == opp:
                _, libs = self._group(nr, nc)
                if len(libs) == 0:
                    captures = True
                    break

        # Check if the placed stone's group has liberties
        _, own_libs = self._group(row, col)

        # Remove the stone
        self.grid[row, col] = EMPTY

        # Suicide if no captures and no liberties
        return not captures and len(own_libs) == 0

    # ── Move Legality ───────────────────────────────────────────────

    def is_legal(self, color: int, move: tuple[int, int] | None) -> bool:
        """
        Check if a move is legal.
        - Pass (None) is always legal.
        - Must be on empty intersection.
        - Must not be suicide.
        - Must not violate positional superko.
        """
        if move is None:
            return True

        row, col = move

        # Bounds check
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False

        # Must be empty
        if self.grid[row, col] != EMPTY:
            return False

        # Suicide check
        if self._is_suicide(color, row, col):
            return False

        # Superko check: simulate the move in-place, check hash, then undo.
        # This avoids a full board copy per intersection.
        self.grid[row, col] = color
        self._toggle_hash(row, col, color)

        # Capture opponent groups with zero liberties
        opp = OPPONENT[color]
        captured_stones: list[tuple[int, int]] = []
        for nr, nc in self._neighbors(row, col):
            if self.grid[nr, nc] == opp:
                stones, liberties = self._group(nr, nc)
                if len(liberties) == 0:
                    for sr, sc in stones:
                        self.grid[sr, sc] = EMPTY
                        self._toggle_hash(sr, sc, opp)
                        captured_stones.append((sr, sc))

        result_hash = int(self._hash)
        is_superko = result_hash in self._position_history

        # Undo: restore captured stones, remove placed stone
        for sr, sc in captured_stones:
            self.grid[sr, sc] = opp
            self._toggle_hash(sr, sc, opp)
        self.grid[row, col] = EMPTY
        self._toggle_hash(row, col, color)

        return not is_superko

    def legal_moves(self, color: int) -> list[tuple[int, int] | None]:
        """
        Return all legal moves for the given color, including pass.
        Pass is always the last element.
        """
        moves = []
        for row in range(self.size):
            for col in range(self.size):
                if self.is_legal(color, (row, col)):
                    moves.append((row, col))
        moves.append(None)  # pass is always legal
        return moves

    # ── Play ────────────────────────────────────────────────────────

    def play(self, color: int, move: tuple[int, int] | None) -> None:
        """
        Play a move. Raises ValueError if illegal.

        Args:
            color: BLACK or WHITE
            move: (row, col) or None for pass
        """
        if not self.is_legal(color, move):
            raise ValueError(f"Illegal move: {move} for {'black' if color == BLACK else 'white'}")

        if move is None:
            self.move_history.append((color, None))
            return

        row, col = move
        captured = self._place_stone(color, row, col)
        self.captured[color] += captured
        self._position_history.add(int(self._hash))
        self.move_history.append((color, (row, col)))

    # ── Scoring (Chinese Rules — Area Scoring) ──────────────────────

    def score(self) -> dict:
        """
        Score the board using Chinese rules (area scoring).
        Territory = empty points surrounded entirely by one color.
        Score = stones on board + territory + komi (for white).

        Returns:
            dict with keys: 'black', 'white', 'winner', 'margin'
        """
        visited = np.zeros((self.size, self.size), dtype=bool)
        territory = {BLACK: 0, WHITE: 0}

        for row in range(self.size):
            for col in range(self.size):
                if visited[row, col] or self.grid[row, col] != EMPTY:
                    continue

                # BFS to find connected empty region
                region = set()
                borders = set()  # colors that border this region
                queue = deque([(row, col)])
                visited[row, col] = True

                while queue:
                    r, c = queue.popleft()
                    region.add((r, c))
                    for nr, nc in self._neighbors(r, c):
                        cell = self.grid[nr, nc]
                        if cell != EMPTY:
                            borders.add(cell)
                        elif not visited[nr, nc]:
                            visited[nr, nc] = True
                            queue.append((nr, nc))

                # Territory belongs to a color only if bordered by exactly one color
                if len(borders) == 1:
                    owner = borders.pop()
                    territory[owner] += len(region)

        # Count stones on board
        black_stones = int(np.sum(self.grid == BLACK))
        white_stones = int(np.sum(self.grid == WHITE))

        black_score = black_stones + territory[BLACK]
        white_score = white_stones + territory[WHITE] + self.komi

        if black_score > white_score:
            winner = "black"
        elif white_score > black_score:
            winner = "white"
        else:
            winner = "draw"

        return {
            "black": black_score,
            "white": white_score,
            "winner": winner,
            "margin": abs(black_score - white_score),
        }

    # ── Display ─────────────────────────────────────────────────────

    def __str__(self) -> str:
        """Pretty-print the board."""
        symbols = {EMPTY: ".", BLACK: "X", WHITE: "O"}
        col_labels = "  " + " ".join(
            chr(ord("A") + i + (1 if i >= 8 else 0)) for i in range(self.size)
        )
        rows = [col_labels]
        for r in range(self.size):
            row_label = f"{self.size - r:2d}"
            row_str = " ".join(symbols[self.grid[r, c]] for c in range(self.size))
            rows.append(f"{row_label} {row_str} {row_label}")
        rows.append(col_labels)
        return "\n".join(rows)
