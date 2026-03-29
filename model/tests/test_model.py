"""
Tests for neural network and feature encoding.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np

from go_engine.board import Board, BLACK, WHITE
from model.features import board_to_features, board_to_tensor, apply_symmetry, NUM_FEATURES
from model.network import AlphaZeroNet, create_network


# ── Feature Encoding Tests ──────────────────────────────────────────


class TestFeatures:
    def test_output_shape(self):
        board = Board(size=9)
        features = board_to_features(board, BLACK)
        assert features.shape == (NUM_FEATURES, 9, 9)
        assert features.dtype == np.float32

    def test_empty_board_features(self):
        board = Board(size=9)
        f = board_to_features(board, BLACK)
        # No stones → planes 0,1 all zeros, plane 2 all ones
        assert f[0].sum() == 0  # current player stones
        assert f[1].sum() == 0  # opponent stones
        assert f[2].sum() == 81  # empty points
        assert f[3].sum() == 0  # no last move
        assert f[4].sum() == 81  # black to play → all 1s
        assert f[5].sum() == 0  # no stones in atari
        assert f[6].sum() == 0
        assert f[7].sum() == 0

    def test_color_to_play(self):
        board = Board(size=9)
        f_black = board_to_features(board, BLACK)
        f_white = board_to_features(board, WHITE)
        assert f_black[4].sum() == 81  # all 1s for black
        assert f_white[4].sum() == 0  # all 0s for white

    def test_stone_planes(self):
        board = Board(size=9)
        board.play(BLACK, (4, 4))
        board.play(WHITE, (0, 0))

        # From black's perspective
        f = board_to_features(board, BLACK)
        assert f[0, 4, 4] == 1.0  # current player (black) stone
        assert f[1, 0, 0] == 1.0  # opponent (white) stone
        assert f[0, 0, 0] == 0.0
        assert f[1, 4, 4] == 0.0

        # From white's perspective: stones swap planes
        f = board_to_features(board, WHITE)
        assert f[0, 0, 0] == 1.0  # current player (white) stone
        assert f[1, 4, 4] == 1.0  # opponent (black) stone

    def test_last_move_plane(self):
        board = Board(size=9)
        board.play(BLACK, (3, 5))
        f = board_to_features(board, WHITE)
        assert f[3, 3, 5] == 1.0
        assert f[3].sum() == 1.0  # only one cell marked

    def test_last_move_pass(self):
        board = Board(size=9)
        board.play(BLACK, None)
        f = board_to_features(board, WHITE)
        assert f[3].sum() == 0.0  # pass → no last move marked

    def test_liberty_features(self):
        board = Board(size=9)
        board.play(BLACK, (0, 0))  # corner stone → 2 liberties
        board.play(BLACK, (4, 4))  # center stone → 4 liberties
        f = board_to_features(board, BLACK)
        assert f[6, 0, 0] == 1.0  # 2 liberties
        assert f[7, 4, 4] == 1.0  # ≥3 liberties

    def test_atari_feature(self):
        """Stone with 1 liberty should be marked in the atari plane."""
        board = Board(size=9)
        board.play(BLACK, (0, 0))
        board.play(WHITE, (0, 1))  # reduces black's liberties to 1
        f = board_to_features(board, BLACK)
        assert f[5, 0, 0] == 1.0  # atari!

    def test_tensor_has_batch_dim(self):
        board = Board(size=9)
        t = board_to_tensor(board, BLACK)
        assert t.shape == (1, NUM_FEATURES, 9, 9)
        assert t.dtype == torch.float32


class TestSymmetry:
    def test_identity(self):
        f = np.random.randn(8, 9, 9).astype(np.float32)
        p = np.random.rand(82).astype(np.float32)
        p /= p.sum()
        f2, p2 = apply_symmetry(f, p, 0, 9)
        np.testing.assert_array_equal(f2, f)
        np.testing.assert_array_almost_equal(p2, p)

    def test_all_8_symmetries_preserve_shape(self):
        f = np.random.randn(8, 9, 9).astype(np.float32)
        p = np.random.rand(82).astype(np.float32)
        p /= p.sum()
        for i in range(8):
            f2, p2 = apply_symmetry(f, p, i, 9)
            assert f2.shape == f.shape
            assert p2.shape == p.shape
            assert abs(p2.sum() - 1.0) < 1e-5

    def test_symmetries_are_distinct(self):
        """All 8 symmetries should produce different results (for non-symmetric input)."""
        np.random.seed(42)
        f = np.random.randn(8, 9, 9).astype(np.float32)
        p = np.random.rand(82).astype(np.float32)
        p /= p.sum()
        results = []
        for i in range(8):
            f2, _ = apply_symmetry(f, p, i, 9)
            results.append(f2.tobytes())
        assert len(set(results)) == 8  # all unique


# ── Network Tests ───────────────────────────────────────────────────


class TestNetwork:
    def test_output_shapes(self):
        net = create_network(board_size=9, num_blocks=2, num_filters=32)
        x = torch.randn(4, NUM_FEATURES, 9, 9)
        log_policy, value = net(x)
        assert log_policy.shape == (4, 82)  # 81 + pass
        assert value.shape == (4, 1)

    def test_policy_sums_to_one(self):
        net = create_network(board_size=9, num_blocks=2, num_filters=32)
        x = torch.randn(1, NUM_FEATURES, 9, 9)
        policy, value = net.predict(x)
        assert abs(policy.sum().item() - 1.0) < 1e-5

    def test_value_in_range(self):
        net = create_network(board_size=9, num_blocks=2, num_filters=32)
        x = torch.randn(10, NUM_FEATURES, 9, 9)
        _, value = net.predict(x)
        assert (value >= -1.0).all()
        assert (value <= 1.0).all()

    def test_parameter_count(self):
        net = create_network(board_size=13, num_blocks=6, num_filters=128)
        params = net.count_parameters()
        # Should be roughly 1-2M parameters
        assert 500_000 < params < 5_000_000, f"Got {params} params"

    def test_13x13_output_shapes(self):
        net = create_network(board_size=13, num_blocks=6, num_filters=128)
        x = torch.randn(2, NUM_FEATURES, 13, 13)
        log_policy, value = net(x)
        assert log_policy.shape == (2, 170)  # 169 + pass
        assert value.shape == (2, 1)

    def test_overfit_single_position(self):
        """
        The network should be able to overfit a single position.
        This verifies the training pipeline works end-to-end.
        """
        net = create_network(board_size=5, num_blocks=2, num_filters=32)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

        # Create a target: move at (2,2) with value +1
        board = Board(size=5)
        x = board_to_tensor(board, BLACK)
        target_policy = torch.zeros(1, 26)  # 25 + pass
        target_policy[0, 12] = 1.0  # move (2,2) = index 2*5+2 = 12
        target_value = torch.tensor([[1.0]])

        # Train for a few steps
        net.train()
        for _ in range(100):
            optimizer.zero_grad()
            log_policy, value = net(x)
            policy_loss = -torch.sum(target_policy * log_policy)
            value_loss = F.mse_loss(value, target_value)
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()

        # Check that it learned
        policy, value = net.predict(x)
        assert policy[0, 12].item() > 0.5, f"Expected policy at (2,2) > 0.5, got {policy[0, 12].item():.3f}"
        assert value[0, 0].item() > 0.5, f"Expected value > 0.5, got {value[0, 0].item():.3f}"
