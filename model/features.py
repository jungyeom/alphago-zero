"""
Board state → tensor encoding for the neural network.

Converts a Board object into a multi-channel tensor that the ResNet can process.
Each channel is a 2D binary (or scalar) plane of size board_size x board_size.

Feature planes:
  0: current player's stones (1 where current player has a stone)
  1: opponent's stones
  2: empty points
  3: last move (one-hot, all zeros if no moves yet or last was pass)
  4: color to play (all 1s if black, all 0s if white)
  5: liberties == 1 (stones in atari)
  6: liberties == 2
  7: liberties >= 3

Total: 8 channels
"""

import numpy as np
import torch

from go_engine.board import Board, BLACK, WHITE, EMPTY, OPPONENT

# Try to import C++ feature extraction
try:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import alphago_core as _ac
    _HAS_CPP_FEATURES = True
except ImportError:
    _HAS_CPP_FEATURES = False


NUM_FEATURES = 8


def board_to_features(board: Board, color_to_play: int) -> np.ndarray:
    """
    Convert a board state to a feature tensor.

    Args:
        board: Current board state
        color_to_play: BLACK or WHITE (whose turn it is)

    Returns:
        numpy array of shape (NUM_FEATURES, board_size, board_size), dtype float32
    """
    size = board.size
    opp = OPPONENT[color_to_play]
    features = np.zeros((NUM_FEATURES, size, size), dtype=np.float32)

    # Plane 0: current player's stones
    features[0] = (board.grid == color_to_play).astype(np.float32)

    # Plane 1: opponent's stones
    features[1] = (board.grid == opp).astype(np.float32)

    # Plane 2: empty points
    features[2] = (board.grid == EMPTY).astype(np.float32)

    # Plane 3: last move (one-hot)
    if board.move_history:
        _, last_move = board.move_history[-1]
        if last_move is not None:
            r, c = last_move
            features[3, r, c] = 1.0

    # Plane 4: color to play (all 1s for black, all 0s for white)
    if color_to_play == BLACK:
        features[4] = 1.0

    # Planes 5-7: liberty features for ALL stones on the board
    # We compute liberties per group and mark each stone in the group
    visited = np.zeros((size, size), dtype=bool)
    for r in range(size):
        for c in range(size):
            if board.grid[r, c] == EMPTY or visited[r, c]:
                continue
            stones, liberties = board._group(r, c)
            lib_count = len(liberties)
            for sr, sc in stones:
                visited[sr, sc] = True
                if lib_count == 1:
                    features[5, sr, sc] = 1.0
                elif lib_count == 2:
                    features[6, sr, sc] = 1.0
                elif lib_count >= 3:
                    features[7, sr, sc] = 1.0

    return features


def board_to_features_cpp(board, color_to_play: int) -> np.ndarray:
    """
    C++ accelerated feature extraction.
    Works with both CppBoard (has ._board) and CBoard directly.
    """
    # Get the underlying C++ board
    cboard = getattr(board, '_board', None)
    if cboard is None and hasattr(board, 'at'):
        # It's already a CBoard
        cboard = board

    if cboard is None:
        # Fall back to Python
        return board_to_features(board, color_to_play)

    # Get last move for plane 3
    last_r, last_c = -1, -1
    if hasattr(board, 'move_history') and board.move_history:
        _, last_move = board.move_history[-1]
        if last_move is not None:
            last_r, last_c = last_move

    return _ac.board_to_features_cpp(cboard, color_to_play, last_r, last_c)


def board_to_tensor(board, color_to_play: int) -> torch.Tensor:
    """
    Convert board state to a PyTorch tensor ready for the network.

    Uses C++ feature extraction when available for ~10x speedup.

    Returns:
        Tensor of shape (1, NUM_FEATURES, board_size, board_size)
        (batch dimension included)
    """
    if _HAS_CPP_FEATURES and hasattr(board, '_board'):
        features = board_to_features_cpp(board, color_to_play)
    else:
        features = board_to_features(board, color_to_play)
    return torch.from_numpy(features).unsqueeze(0)


def apply_symmetry(features: np.ndarray, policy: np.ndarray, sym_index: int, board_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply one of 8 board symmetries (4 rotations x 2 reflections) to both
    feature planes and the policy vector.

    This gives us 8x training data for free since Go is symmetric.

    Args:
        features: shape (C, size, size)
        policy: shape (size*size + 1,) — board moves + pass
        sym_index: 0-7 (0 = identity)
        board_size: size of the board

    Returns:
        (transformed_features, transformed_policy)
    """
    # Separate pass probability from board probabilities
    board_policy = policy[:board_size * board_size].reshape(board_size, board_size)
    pass_prob = policy[-1]

    # Apply symmetry to feature planes and policy board
    f = features.copy()
    p = board_policy.copy()

    if sym_index >= 4:
        # Reflect horizontally
        f = f[:, :, ::-1].copy()
        p = p[:, ::-1].copy()

    rotations = sym_index % 4
    if rotations > 0:
        f = np.rot90(f, k=rotations, axes=(1, 2)).copy()
        p = np.rot90(p, k=rotations).copy()

    # Reconstruct policy vector
    transformed_policy = np.append(p.flatten(), pass_prob)

    return f, transformed_policy
