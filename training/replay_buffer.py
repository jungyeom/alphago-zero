"""
Replay buffer for storing self-play training data.

Each entry is a position from a self-play game:
  (features, policy_target, value_target)

Uses pre-allocated numpy arrays to avoid memory fragmentation
from thousands of individual array objects.
"""

import numpy as np


class ReplayBuffer:
    """
    Fixed-size ring buffer that stores self-play positions
    in pre-allocated contiguous numpy arrays.

    Stores:
      - features: board state tensor (C, H, W)
      - policy: MCTS visit distribution (board_size² + 1,)
      - value: game outcome from current player's perspective (-1 or +1)
    """

    def __init__(self, max_size: int = 50_000, board_size: int = 13, num_features: int = 8):
        self.max_size = max_size
        self.board_size = board_size
        self.num_features = num_features
        self.num_moves = board_size * board_size + 1

        # Pre-allocate contiguous arrays
        self._features = np.zeros((max_size, num_features, board_size, board_size), dtype=np.float32)
        self._policies = np.zeros((max_size, self.num_moves), dtype=np.float32)
        self._values = np.zeros(max_size, dtype=np.float32)

        self._size = 0       # current number of valid entries
        self._index = 0      # next write position (ring buffer)

    def push(self, features: np.ndarray, policy: np.ndarray, value: float) -> None:
        """Add a single position to the buffer."""
        self._features[self._index] = features
        self._policies[self._index] = policy
        self._values[self._index] = value

        self._index = (self._index + 1) % self.max_size
        self._size = min(self._size + 1, self.max_size)

    def push_game(
        self,
        features_list: list[np.ndarray],
        policies_list: list[np.ndarray],
        colors_list: list[int],
        game_result: float,
        board_size: int,
        use_symmetry: bool = True,
    ) -> int:
        """
        Add all positions from a completed game.

        Args:
            features_list: list of feature tensors (C, H, W) for each move
            policies_list: list of MCTS policy vectors for each move
            colors_list: list of colors (who played) for each move
            game_result: +1 if black won, -1 if white won
            board_size: size of the board
            use_symmetry: whether to apply 8-fold symmetry augmentation

        Returns:
            Number of positions added
        """
        from model.features import apply_symmetry
        from go_engine.board import BLACK

        count = 0
        sym_range = 8 if use_symmetry else 1

        for features, policy, color in zip(features_list, policies_list, colors_list):
            # Value from current player's perspective
            if color == BLACK:
                value = game_result  # +1 if black won
            else:
                value = -game_result  # flip for white

            for sym_idx in range(sym_range):
                if sym_idx == 0:
                    f, p = features, policy
                else:
                    f, p = apply_symmetry(features, policy, sym_idx, board_size)

                self.push(f, p, value)
                count += 1

        return count

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random batch from the buffer.

        Returns:
            Tuple of (features, policies, values) as numpy arrays.
            - features: (batch, C, H, W)
            - policies: (batch, num_moves)
            - values: (batch, 1)
        """
        indices = np.random.choice(self._size, size=min(batch_size, self._size), replace=False)

        # Slicing pre-allocated arrays — no new allocations
        batch_features = self._features[indices]
        batch_policies = self._policies[indices]
        batch_values = self._values[indices].reshape(-1, 1)

        return batch_features, batch_policies, batch_values

    def __len__(self) -> int:
        return self._size

    def clear(self) -> None:
        self._size = 0
        self._index = 0
