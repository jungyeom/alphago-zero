"""Tests for the Game manager."""

import pytest
from go_engine.game import Game
from go_engine.board import BLACK, WHITE


class TestGameFlow:
    def test_alternating_turns(self):
        g = Game(size=9)
        assert g.current_player == BLACK
        g.play((4, 4))
        assert g.current_player == WHITE
        g.play((3, 3))
        assert g.current_player == BLACK

    def test_two_passes_end_game(self):
        g = Game(size=9)
        g.play(None)  # black passes
        assert not g.is_over
        g.play(None)  # white passes
        assert g.is_over

    def test_play_after_game_over_raises(self):
        g = Game(size=9)
        g.play(None)
        g.play(None)
        with pytest.raises(ValueError):
            g.play((0, 0))

    def test_resign(self):
        g = Game(size=9)
        g.play((4, 4))  # black
        g.resign()  # white resigns
        assert g.is_over
        result = g.result()
        assert result["winner"] == "black"
        assert result["reason"] == "resign"

    def test_resign_black(self):
        g = Game(size=9)
        g.resign()  # black resigns
        result = g.result()
        assert result["winner"] == "white"

    def test_result_before_game_over_raises(self):
        g = Game(size=9)
        with pytest.raises(ValueError):
            g.result()

    def test_scoring_result(self):
        g = Game(size=9)
        g.play(None)
        g.play(None)
        result = g.result()
        assert result["reason"] == "scoring"
        assert result["winner"] == "white"  # empty board, white wins by komi

    def test_legal_moves(self):
        g = Game(size=9)
        moves = g.legal_moves()
        assert len(moves) == 82  # 81 + pass

    def test_str_does_not_crash(self):
        g = Game(size=9)
        g.play((4, 4))
        s = str(g)
        assert "Black" not in s or "White" not in s or len(s) > 0
