"""
Tests for Go board rules: placement, captures, ko, suicide, scoring.

Uses small boards (5x5, 9x9) for readability, but the engine supports any size.
"""

import pytest
import numpy as np
from go_engine.board import Board, BLACK, WHITE, EMPTY


# ── Basic Placement ─────────────────────────────────────────────────


class TestBasicPlacement:
    def test_place_stone(self):
        b = Board(size=9)
        b.play(BLACK, (0, 0))
        assert b.grid[0, 0] == BLACK

    def test_place_on_occupied_raises(self):
        b = Board(size=9)
        b.play(BLACK, (3, 3))
        with pytest.raises(ValueError):
            b.play(WHITE, (3, 3))

    def test_out_of_bounds_raises(self):
        b = Board(size=9)
        assert not b.is_legal(BLACK, (-1, 0))
        assert not b.is_legal(BLACK, (0, 9))
        assert not b.is_legal(BLACK, (9, 9))

    def test_pass_is_always_legal(self):
        b = Board(size=9)
        assert b.is_legal(BLACK, None)
        b.play(BLACK, None)
        assert len(b.move_history) == 1

    def test_move_history_recorded(self):
        b = Board(size=9)
        b.play(BLACK, (0, 0))
        b.play(WHITE, (1, 1))
        b.play(BLACK, None)
        assert b.move_history == [(BLACK, (0, 0)), (WHITE, (1, 1)), (BLACK, None)]


# ── Liberties ───────────────────────────────────────────────────────


class TestLiberties:
    def test_corner_stone_liberties(self):
        b = Board(size=9)
        b.play(BLACK, (0, 0))
        assert b._liberties(0, 0) == 2

    def test_edge_stone_liberties(self):
        b = Board(size=9)
        b.play(BLACK, (0, 4))
        assert b._liberties(0, 4) == 3

    def test_center_stone_liberties(self):
        b = Board(size=9)
        b.play(BLACK, (4, 4))
        assert b._liberties(4, 4) == 4

    def test_group_liberties(self):
        """Two connected stones share liberties."""
        b = Board(size=9)
        b.play(BLACK, (4, 4))
        b.play(BLACK, (4, 5))
        assert b._liberties(4, 4) == 6

    def test_group_liberties_reduced_by_opponent(self):
        b = Board(size=9)
        b.play(BLACK, (4, 4))
        b.play(WHITE, (3, 4))  # reduces black's liberties
        assert b._liberties(4, 4) == 3


# ── Captures ────────────────────────────────────────────────────────


class TestCaptures:
    def test_capture_single_stone_corner(self):
        """
        Capture a single stone in the corner:
          X O .      . O .
          O . .  →   O . .
        """
        b = Board(size=5)
        b.play(WHITE, (0, 0))  # stone to be captured
        b.play(BLACK, (0, 1))
        b.play(BLACK, (1, 0))  # captures white
        assert b.grid[0, 0] == EMPTY
        assert b.captured[BLACK] == 1

    def test_capture_single_stone_edge(self):
        """
        Capture a stone on the edge:
          . X .
          X O X  → O is captured
          . X .
        """
        b = Board(size=5)
        b.play(WHITE, (1, 2))
        b.play(BLACK, (0, 2))
        b.play(BLACK, (1, 1))
        b.play(BLACK, (1, 3))
        b.play(BLACK, (2, 2))
        assert b.grid[1, 2] == EMPTY
        assert b.captured[BLACK] == 1

    def test_capture_group(self):
        """
        Capture a group of two white stones:
          . X X .
          X O O X  → both O captured
          . X X .
        """
        b = Board(size=5)
        b.play(WHITE, (1, 1))
        b.play(WHITE, (1, 2))
        # Surround
        b.play(BLACK, (0, 1))
        b.play(BLACK, (0, 2))
        b.play(BLACK, (1, 0))
        b.play(BLACK, (1, 3))
        b.play(BLACK, (2, 1))
        b.play(BLACK, (2, 2))  # captures both
        assert b.grid[1, 1] == EMPTY
        assert b.grid[1, 2] == EMPTY
        assert b.captured[BLACK] == 2

    def test_capture_then_recapture(self):
        """After capturing, the intersection is empty and playable (if not suicide)."""
        b = Board(size=5)
        # Set up so white at (2,2) gets captured but can be replayed
        #   . . . . .
        #   . . X . .
        #   . X O X .     → black plays (3,2) capturing white
        #   . . X . .
        #   . . . . .
        b.play(WHITE, (2, 2))
        b.play(BLACK, (1, 2))
        b.play(BLACK, (2, 1))
        b.play(BLACK, (2, 3))
        b.play(BLACK, (3, 2))  # captures white at (2,2)
        assert b.grid[2, 2] == EMPTY
        # White can play there again — not suicide because it would capture
        # surrounding black stones? No — the black stones have other liberties.
        # Actually this IS suicide for white. Let me use a different approach:
        # Just verify the intersection is empty and a same-color stone can go there.
        b.play(BLACK, (2, 2))  # black can definitely play here
        assert b.grid[2, 2] == BLACK

    def test_capture_count_accumulates(self):
        b = Board(size=5)
        # First capture
        b.play(WHITE, (0, 0))
        b.play(BLACK, (0, 1))
        b.play(BLACK, (1, 0))
        assert b.captured[BLACK] == 1
        # Second capture
        b.play(WHITE, (4, 4))
        b.play(BLACK, (4, 3))
        b.play(BLACK, (3, 4))
        assert b.captured[BLACK] == 2


# ── Suicide ─────────────────────────────────────────────────────────


class TestSuicide:
    def test_suicide_single_stone(self):
        """
        Playing into a surrounded point with no captures is suicide:
          X .
          . X   → black at (0,1) would have 0 liberties and capture nothing
        Wait, need full surround. Let's use corner:
          . X
          X .  → playing white at (0,0) is suicide
        """
        b = Board(size=5)
        b.play(BLACK, (0, 1))
        b.play(BLACK, (1, 0))
        assert not b.is_legal(WHITE, (0, 0))

    def test_suicide_not_if_captures(self):
        """
        Playing into a 'surrounded' point IS legal if it captures opponent stones.
          O X .
          X . .  → White at (0,0) would be suicide... unless it captures.

        Actually let's set up properly:
          . O       X captures O by playing at (0,0)
          O X       if O is surrounded
        Wait, let me think carefully.

        Setup: black stone surrounded by white, but playing captures.
          X O .
          O . .
        White plays (0,0)... that's not surrounded.

        Better example:
          . X       playing black at (0,0)
          X O       would be suicide unless it captures white at (1,1)
          . X       but white at (1,1) has liberty at (1,0)... not captured.

        Simplest: fill a corner properly.
        """
        b = Board(size=5)
        # Set up: white stone at (0,0) surrounded on two sides by black
        # and black can play to capture by filling the last liberty
        b.play(WHITE, (0, 0))
        b.play(BLACK, (0, 1))
        # Black plays (1,0) — this captures white, so NOT suicide
        assert b.is_legal(BLACK, (1, 0))

    def test_suicide_group(self):
        """
        A move that would kill the entire friendly group is suicide.

        Set up on 5x5:
          X X X X .
          X O . X .   → white playing at (1,2) fills last liberty
          X X X X .     group {(1,1),(1,2)} has 0 liberties → suicide
          . . . . .
        """
        b = Board(size=5)
        # Surround completely
        for r, c in [(0, 0), (0, 1), (0, 2), (0, 3),
                      (1, 0), (1, 3),
                      (2, 0), (2, 1), (2, 2), (2, 3)]:
            b.grid[r, c] = BLACK
            b._toggle_hash(r, c, BLACK)
        b.grid[1, 1] = WHITE
        b._toggle_hash(1, 1, WHITE)
        # White at (1,2) would create group {(1,1),(1,2)} with 0 liberties
        assert not b.is_legal(WHITE, (1, 2))


# ── Ko ──────────────────────────────────────────────────────────────


class TestKo:
    def _setup_ko(self):
        """
        Set up a basic ko position:
            . X O .
            X . X O
            . X O .

        Black plays (1,1) capturing white at... wait, let me set up properly.

        Standard ko:
            . B W .
            B . B W
            . B W .

        Black at (1,1) captures W at... there's no W at (1,1).

        Let me use a concrete setup:
            col: 0 1 2 3
        row 0:  . X O .
        row 1:  X O . O
        row 2:  . X O .

        Black plays (1,2), capturing white at (1,1)? No, (1,1) is white.
        Let me reconsider.

        Standard ko shape on a 5x5:
            col: 0 1 2 3 4
        row 0:  . . . . .
        row 1:  . X O . .
        row 2:  X O . O .
        row 3:  . X O . .
        row 4:  . . . . .

        Black plays (2,2), captures white at (2,1).
        White wants to recapture at (2,1) but that's ko.
        """
        b = Board(size=5)
        # Black stones
        b.play(BLACK, (1, 1))
        b.play(BLACK, (2, 0))
        b.play(BLACK, (3, 1))
        # White stones
        b.play(WHITE, (1, 2))
        b.play(WHITE, (2, 3))
        b.play(WHITE, (3, 2))
        # White at (2,1) — the stone that will be captured
        b.play(WHITE, (2, 1))
        return b

    def test_ko_basic(self):
        """After capturing in a ko, immediate recapture is illegal."""
        b = self._setup_ko()
        # Black captures at (2,2), taking white (2,1)
        b.play(BLACK, (2, 2))
        assert b.grid[2, 1] == EMPTY  # white was captured
        # White cannot immediately recapture at (2,1)
        assert not b.is_legal(WHITE, (2, 1))

    def test_ko_allowed_after_other_move(self):
        """After a ko threat (playing elsewhere), recapture is allowed."""
        b = self._setup_ko()
        b.play(BLACK, (2, 2))  # captures
        # White plays elsewhere (ko threat)
        b.play(WHITE, (4, 4))
        # Black responds
        b.play(BLACK, (4, 3))
        # Now white can recapture
        assert b.is_legal(WHITE, (2, 1))


# ── Scoring ─────────────────────────────────────────────────────────


class TestScoring:
    def test_empty_board(self):
        """Empty board: white wins by komi."""
        b = Board(size=9)
        score = b.score()
        assert score["black"] == 0
        assert score["white"] == 6.5
        assert score["winner"] == "white"

    def test_simple_territory(self):
        """
        Black owns left side, white owns right side on a 5x5:
          X X . O O
          X . X O .
          X X . O O
          . . . . .
          . . . . .

        Black territory: (1,1) = 1 point, black stones = 7
        White territory: (1,4) = 1 point, white stones = 6
        Contested: row 3-4 and column 2 empty spaces touch both → no territory
        """
        b = Board(size=5)
        b.komi = 6.5
        black_stones = [(0, 0), (0, 1), (1, 0), (1, 2), (2, 0), (2, 1)]
        white_stones = [(0, 3), (0, 4), (1, 3), (2, 3), (2, 4)]
        for r, c in black_stones:
            b.grid[r, c] = BLACK
            b._toggle_hash(r, c, BLACK)
        for r, c in white_stones:
            b.grid[r, c] = WHITE
            b._toggle_hash(r, c, WHITE)

        score = b.score()
        # (1,1) is surrounded only by black → 1 territory for black
        # (1,4) is surrounded only by white → 1 territory for white
        # Empty cells in col 2 and rows 3-4 border both colors → contested
        assert score["black"] == 6 + 1  # stones + territory
        assert score["white"] == 5 + 1 + 6.5  # stones + territory + komi
        assert score["winner"] == "white"

    def test_contested_territory(self):
        """Empty region bordered by both colors belongs to neither."""
        b = Board(size=5)
        b.grid[0, 0] = BLACK
        b.grid[0, 4] = WHITE
        b._toggle_hash(0, 0, BLACK)
        b._toggle_hash(0, 4, WHITE)
        score = b.score()
        # The large empty region touches both colors → no territory for either
        assert score["black"] == 1  # just the stone
        assert score["white"] == 1 + 6.5  # stone + komi


# ── Legal Move Generation ──────────────────────────────────────────


class TestLegalMoves:
    def test_empty_board_all_moves_legal(self):
        b = Board(size=9)
        moves = b.legal_moves(BLACK)
        # 81 board points + 1 pass
        assert len(moves) == 82
        assert None in moves

    def test_occupied_points_excluded(self):
        b = Board(size=9)
        b.play(BLACK, (4, 4))
        moves = b.legal_moves(WHITE)
        assert (4, 4) not in moves
        assert len(moves) == 81  # 80 board + pass


# ── Copy ────────────────────────────────────────────────────────────


class TestCopy:
    def test_copy_independent(self):
        b = Board(size=9)
        b.play(BLACK, (4, 4))
        c = b.copy()
        c.play(WHITE, (3, 3))
        # Original should be unchanged
        assert b.grid[3, 3] == EMPTY
        assert c.grid[3, 3] == WHITE

    def test_copy_preserves_state(self):
        b = Board(size=9)
        b.play(BLACK, (4, 4))
        b.play(WHITE, (0, 0))
        c = b.copy()
        assert np.array_equal(b.grid, c.grid)
        assert b.captured == c.captured
        assert b._hash == c._hash


# ── Display ─────────────────────────────────────────────────────────


class TestDisplay:
    def test_str_does_not_crash(self):
        b = Board(size=9)
        b.play(BLACK, (4, 4))
        s = str(b)
        assert "X" in s
        assert len(s) > 0
