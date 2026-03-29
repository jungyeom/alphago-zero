"""
Game manager — wraps Board with turn tracking, game-end detection, and resign.
"""

from go_engine.board import Board, BLACK, WHITE, OPPONENT


class Game:
    """
    Manages a full game of Go.

    Tracks whose turn it is, detects game end (two consecutive passes),
    handles resign, and provides the final result.

    Usage:
        game = Game(size=13)
        game.play((3, 4))          # black plays
        game.play((2, 3))          # white plays
        game.play(None)            # black passes
        game.play(None)            # white passes → game ends
        result = game.result()     # scoring
    """

    def __init__(self, size: int = 13, komi: float = 6.5):
        self.board = Board(size=size)
        self.board.komi = komi
        self.current_player = BLACK
        self.is_over = False
        self._resigned_by: int | None = None
        self._consecutive_passes = 0

    @property
    def size(self) -> int:
        return self.board.size

    def play(self, move: tuple[int, int] | None) -> None:
        """
        Play a move for the current player.

        Args:
            move: (row, col) or None for pass.
                  Use resign() to resign instead.

        Raises:
            ValueError: if the game is over or the move is illegal.
        """
        if self.is_over:
            raise ValueError("Game is already over")

        self.board.play(self.current_player, move)

        if move is None:
            self._consecutive_passes += 1
        else:
            self._consecutive_passes = 0

        # Two consecutive passes end the game
        if self._consecutive_passes >= 2:
            self.is_over = True

        self.current_player = OPPONENT[self.current_player]

    def resign(self) -> None:
        """Current player resigns."""
        if self.is_over:
            raise ValueError("Game is already over")
        self._resigned_by = self.current_player
        self.is_over = True

    def legal_moves(self) -> list[tuple[int, int] | None]:
        """Legal moves for the current player."""
        return self.board.legal_moves(self.current_player)

    def result(self) -> dict:
        """
        Get the game result.

        Returns:
            dict with: 'black', 'white', 'winner', 'margin', 'reason'
        """
        if not self.is_over:
            raise ValueError("Game is not over yet")

        if self._resigned_by is not None:
            winner = "white" if self._resigned_by == BLACK else "black"
            return {
                "black": 0,
                "white": 0,
                "winner": winner,
                "margin": 0,
                "reason": "resign",
            }

        score = self.board.score()
        score["reason"] = "scoring"
        return score

    def __str__(self) -> str:
        turn = "Black" if self.current_player == BLACK else "White"
        status = "GAME OVER" if self.is_over else f"{turn} to play"
        captures = f"Captures — Black: {self.board.captured[BLACK]}, White: {self.board.captured[WHITE]}"
        return f"{self.board}\n{status} | {captures}"
