"""
End-to-end tests for multi-threaded C++ MCTS with virtual loss.
"""

import time
import pytest
import torch
import numpy as np

from go_engine.board import Board, BLACK, WHITE
from model.network import create_network

try:
    import alphago_core as ac
    from mcts.cpp_search import (
        search_cpp,
        search_parallel_cpp,
        get_move_probabilities_parallel_cpp,
        _make_batch_eval_fn,
    )
    HAS_CPP = True
except (ImportError, RuntimeError):
    HAS_CPP = False

pytestmark = pytest.mark.skipif(not HAS_CPP, reason="C++ module not available")

BOARD_SIZE = 9
DEVICE = torch.device("cpu")


@pytest.fixture(scope="module")
def net():
    model = create_network(board_size=BOARD_SIZE, num_blocks=2, num_filters=32)
    model.eval()
    return model


@pytest.fixture
def empty_board():
    return Board(size=BOARD_SIZE)


class TestSearchParallelCompletes:
    """Verify search_parallel runs to completion without deadlocks."""

    @pytest.mark.timeout(30)
    def test_basic_completion(self, net, empty_board):
        move, policy = search_parallel_cpp(
            empty_board, BLACK, net,
            num_simulations=50, num_threads=2,
            temperature=1.0, device=DEVICE,
        )
        assert policy.shape == (BOARD_SIZE * BOARD_SIZE + 1,)
        assert abs(policy.sum() - 1.0) < 0.01
        # Move should be legal (a coordinate tuple or None for pass)
        if move is not None:
            r, c = move
            assert 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

    @pytest.mark.timeout(30)
    @pytest.mark.parametrize("num_threads", [1, 2, 4])
    def test_multiple_thread_counts(self, net, empty_board, num_threads):
        move, policy = search_parallel_cpp(
            empty_board, BLACK, net,
            num_simulations=50, num_threads=num_threads,
            temperature=1.0, device=DEVICE,
        )
        assert policy.shape == (BOARD_SIZE * BOARD_SIZE + 1,)
        assert abs(policy.sum() - 1.0) < 0.01

    @pytest.mark.timeout(30)
    @pytest.mark.parametrize("min_bs,max_bs", [(1, 1), (1, 4), (4, 8)])
    def test_batch_size_configs(self, net, empty_board, min_bs, max_bs):
        move, policy = search_parallel_cpp(
            empty_board, BLACK, net,
            num_simulations=50, num_threads=2,
            min_batch_size=min_bs, max_batch_size=max_bs,
            temperature=1.0, device=DEVICE,
        )
        assert policy.shape == (BOARD_SIZE * BOARD_SIZE + 1,)
        assert abs(policy.sum() - 1.0) < 0.01


class TestSearchParallelMidGame:
    """Verify parallel search works on non-empty board positions."""

    @pytest.mark.timeout(30)
    def test_mid_game_position(self, net):
        board = Board(size=BOARD_SIZE)
        # Play a few moves
        moves = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]
        color = BLACK
        for r, c in moves:
            board.play(color, (r, c))
            color = WHITE if color == BLACK else BLACK

        move, policy = search_parallel_cpp(
            board, color, net,
            num_simulations=50, num_threads=2,
            temperature=1.0, device=DEVICE,
        )
        assert policy.shape == (BOARD_SIZE * BOARD_SIZE + 1,)
        assert abs(policy.sum() - 1.0) < 0.01

    @pytest.mark.timeout(30)
    def test_white_to_play(self, net):
        board = Board(size=BOARD_SIZE)
        board.play(BLACK, (4, 4))

        move, policy = search_parallel_cpp(
            board, WHITE, net,
            num_simulations=50, num_threads=2,
            temperature=1.0, device=DEVICE,
        )
        assert policy.shape == (BOARD_SIZE * BOARD_SIZE + 1,)
        assert abs(policy.sum() - 1.0) < 0.01


class TestSearchParallelConsistency:
    """Compare single-threaded and parallel search produce valid outputs."""

    @pytest.mark.timeout(30)
    def test_both_produce_valid_policies(self, net, empty_board):
        move_st, policy_st = search_cpp(
            empty_board, BLACK, net,
            num_simulations=50,
            temperature=1.0, device=DEVICE,
        )
        move_mt, policy_mt = search_parallel_cpp(
            empty_board, BLACK, net,
            num_simulations=50, num_threads=2,
            temperature=1.0, device=DEVICE,
        )

        # Both should produce valid policy distributions
        assert policy_st.shape == policy_mt.shape
        assert abs(policy_st.sum() - 1.0) < 0.01
        assert abs(policy_mt.sum() - 1.0) < 0.01

        # Both moves should be legal
        for move in [move_st, move_mt]:
            if move is not None:
                r, c = move
                assert 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


class TestMultiThreadUtilization:
    """Verify that multiple threads actually do work concurrently."""

    def _run_parallel_raw(self, net, num_threads, num_sims=200):
        """Run search_parallel via the C API to get ParallelStats."""
        cboard = ac.CBoard(BOARD_SIZE)
        batch_eval_fn = _make_batch_eval_fn(net, BOARD_SIZE, DEVICE)

        config = ac.MCTSConfig()
        config.num_simulations = num_sims
        config.num_threads = num_threads
        config.min_batch_size = 1
        config.max_batch_size = 8

        mcts = ac.MCTSSearch(config)
        return mcts.search_parallel(cboard, 1, batch_eval_fn, 1.0)

    @pytest.mark.timeout(30)
    def test_work_distributed_across_threads(self, net):
        """Each thread should complete a share of simulations."""
        for num_threads in [2, 4]:
            result = self._run_parallel_raw(net, num_threads, num_sims=200)
            stats = result.parallel_stats
            sims = list(stats.sims_per_thread)

            assert len(sims) == num_threads
            assert sum(sims) == 200
            # Each thread should do meaningful work (at least 10% of total)
            for s in sims:
                assert s >= 20, f"Thread only did {s}/200 sims — not enough work"

    @pytest.mark.timeout(30)
    def test_batching_occurs(self, net):
        """With 4 threads, evaluator should batch requests together."""
        result = self._run_parallel_raw(net, num_threads=4, num_sims=200)
        stats = result.parallel_stats

        assert stats.total_batch_items == 200
        # With 4 threads and max_batch_size=8, we should get fewer batches
        # than total items (i.e., actual batching happened)
        assert stats.num_batches < stats.total_batch_items, \
            f"No batching: {stats.num_batches} batches for {stats.total_batch_items} items"

    @pytest.mark.timeout(30)
    def test_single_thread_no_sharing(self, net):
        """With 1 thread, all sims should be on that thread."""
        result = self._run_parallel_raw(net, num_threads=1, num_sims=100)
        stats = result.parallel_stats
        assert list(stats.sims_per_thread) == [100]

    @pytest.mark.timeout(60)
    def test_speedup_over_single_threaded(self, net):
        """Multi-threaded search should be faster than single-threaded."""
        board = Board(size=BOARD_SIZE)
        num_sims = 200

        # Single-threaded timing
        times_st = []
        for _ in range(3):
            t0 = time.perf_counter()
            search_cpp(board, BLACK, net, num_simulations=num_sims, device=DEVICE)
            times_st.append(time.perf_counter() - t0)
        avg_st = sum(times_st) / len(times_st)

        # 4-thread timing
        times_mt = []
        for _ in range(3):
            t0 = time.perf_counter()
            search_parallel_cpp(
                board, BLACK, net, num_simulations=num_sims,
                num_threads=4, device=DEVICE,
            )
            times_mt.append(time.perf_counter() - t0)
        avg_mt = sum(times_mt) / len(times_mt)

        speedup = avg_st / avg_mt
        # Expect at least 1.3x speedup even on CPU
        assert speedup > 1.3, \
            f"No meaningful speedup: {avg_st:.3f}s vs {avg_mt:.3f}s ({speedup:.2f}x)"


class TestGetMoveProbabilitiesParallel:
    """Test the higher-level wrapper function."""

    @pytest.mark.timeout(30)
    def test_returns_correct_format(self, net, empty_board):
        move_probs, policy_vec = get_move_probabilities_parallel_cpp(
            empty_board, BLACK, net,
            num_simulations=50, num_threads=2,
            temperature=1.0, device=DEVICE,
        )
        assert isinstance(move_probs, list)
        assert len(move_probs) > 0
        assert policy_vec.shape == (BOARD_SIZE * BOARD_SIZE + 1,)

        # Each entry is ((row, col), prob) or (None, prob)
        for move, prob in move_probs:
            assert prob > 0
            if move is not None:
                assert len(move) == 2
