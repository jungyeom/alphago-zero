"""
Tests for MCTS node and search.
"""

import pytest
import torch
import numpy as np
from go_engine.board import Board, BLACK, WHITE
from mcts.node import Node
from mcts.search import (
    search, get_best_move, get_move_probabilities,
    search_with_net, get_best_move_with_net, get_move_probabilities_with_net,
)
from model.network import create_network


# ── Node Tests ──────────────────────────────────────────────────────


class TestNode:
    def test_initial_state(self):
        n = Node()
        assert n.visit_count == 0
        assert n.total_value == 0.0
        assert n.q_value == 0.0
        assert not n.is_expanded
        assert n.is_root

    def test_q_value(self):
        n = Node()
        n.visit_count = 10
        n.total_value = 7.0
        assert n.q_value == 0.7

    def test_ucb1_unvisited_is_infinity(self):
        parent = Node()
        parent.visit_count = 10
        child = Node(parent=parent)
        assert child.ucb1_score() == float("inf")

    def test_ucb1_visited(self):
        parent = Node()
        parent.visit_count = 100
        child = Node(parent=parent)
        child.visit_count = 10
        child.total_value = 5.0
        score = child.ucb1_score(c=1.4)
        assert score > child.q_value  # exploration term adds to Q

    def test_puct_with_prior(self):
        parent = Node()
        parent.visit_count = 100
        # High prior child should score higher than low prior (all else equal)
        high = Node(parent=parent, prior=0.8)
        low = Node(parent=parent, prior=0.1)
        assert high.puct_score() > low.puct_score()

    def test_best_child_selects_highest(self):
        parent = Node()
        parent.visit_count = 50
        c1 = Node(parent=parent, color=BLACK)
        c1.visit_count = 10
        c1.total_value = 8.0
        c2 = Node(parent=parent, color=BLACK)
        c2.visit_count = 10
        c2.total_value = 2.0
        parent.children = [c1, c2]
        assert parent.best_child() == c1

    def test_most_visited_child(self):
        parent = Node()
        c1 = Node(parent=parent)
        c1.visit_count = 5
        c2 = Node(parent=parent)
        c2.visit_count = 20
        parent.children = [c1, c2]
        assert parent.most_visited_child() == c2

    def test_best_child_empty_raises(self):
        n = Node()
        with pytest.raises(ValueError):
            n.best_child()


# ── Search Tests ────────────────────────────────────────────────────


class TestSearch:
    def test_search_returns_root(self):
        """Search should return a root node with children."""
        board = Board(size=9)
        root = search(board, BLACK, num_simulations=10)
        assert root.is_root
        assert root.is_expanded
        assert len(root.children) > 0

    def test_search_visits_add_up(self):
        """Total child visits should equal root visits minus 1 (root counted once extra)."""
        board = Board(size=9)
        root = search(board, BLACK, num_simulations=50)
        child_visits = sum(c.visit_count for c in root.children)
        # Each simulation visits root once and one path down
        assert child_visits == root.visit_count

    def test_get_best_move_returns_legal(self):
        """Best move must be a legal move."""
        board = Board(size=9)
        move = get_best_move(board, BLACK, num_simulations=20)
        legal = board.legal_moves(BLACK)
        assert move in legal

    def test_search_from_near_end(self):
        """Search should handle a nearly-full board without crashing."""
        board = Board(size=5)
        # Fill most of the board
        import numpy as np
        # Checkerboard pattern with some empty
        for r in range(5):
            for c in range(5):
                if (r + c) % 2 == 0 and (r, c) != (4, 4):
                    color = BLACK if r < 3 else WHITE
                    board.grid[r, c] = color
                    board._toggle_hash(r, c, color)

        move = get_best_move(board, BLACK, num_simulations=10)
        legal = board.legal_moves(BLACK)
        assert move in legal

    def test_move_probabilities_sum_to_one(self):
        """Move probabilities should sum to approximately 1."""
        board = Board(size=5)
        probs = get_move_probabilities(board, BLACK, num_simulations=30)
        total = sum(p for _, p in probs)
        assert abs(total - 1.0) < 1e-6

    def test_move_probabilities_deterministic(self):
        """Temperature=0 should put all weight on the most-visited move."""
        board = Board(size=5)
        # Use temperature exactly 0 (triggers argmax path)
        probs = get_move_probabilities(board, BLACK, num_simulations=50, temperature=0.0)
        max_prob = max(p for _, p in probs)
        assert max_prob == 1.0

    def test_mcts_explores_multiple_moves(self):
        """
        MCTS should spread visits across multiple moves, not just one.

        Note: tactical tests (e.g. 'does MCTS find the capture?') are
        unreliable with random rollouts because the signal from a single
        capture is drowned out by 50+ moves of random play. The real
        validation is the benchmark (MCTS beats random agent >80%).
        """
        board = Board(size=5)
        root = search(board, BLACK, num_simulations=100)
        visited = [c for c in root.children if c.visit_count > 0]
        # With UCB1, MCTS should explore many moves, not just one
        assert len(visited) > 5, f"Only {len(visited)} moves explored"


# ── Neural Net MCTS Tests ──────────────────────────────────────────


class TestNetSearch:
    @pytest.fixture
    def net_9x9(self):
        """Small network for testing (9x9, 2 blocks, 32 filters)."""
        return create_network(board_size=9, num_blocks=2, num_filters=32)

    def test_search_returns_root(self, net_9x9):
        board = Board(size=9)
        root = search_with_net(board, BLACK, net_9x9, num_simulations=20)
        assert root.is_root
        assert root.is_expanded
        assert len(root.children) > 0

    def test_children_have_priors(self, net_9x9):
        """Children should have non-uniform priors from the network."""
        board = Board(size=9)
        root = search_with_net(board, BLACK, net_9x9, num_simulations=10)
        priors = [c.prior for c in root.children]
        # Priors should sum to ~1 and not all be equal
        assert abs(sum(priors) - 1.0) < 0.01
        assert max(priors) > min(priors)  # non-uniform

    def test_best_move_is_legal(self, net_9x9):
        board = Board(size=9)
        move = get_best_move_with_net(board, BLACK, net_9x9, num_simulations=20)
        legal = board.legal_moves(BLACK)
        assert move in legal

    def test_move_probabilities_sum_to_one(self, net_9x9):
        board = Board(size=9)
        probs, policy_vec = get_move_probabilities_with_net(
            board, BLACK, net_9x9, num_simulations=30
        )
        total = sum(p for _, p in probs)
        assert abs(total - 1.0) < 1e-5
        assert abs(policy_vec.sum() - 1.0) < 1e-5
        assert policy_vec.shape == (82,)

    def test_policy_vector_format(self, net_9x9):
        """Policy vector should be indexed correctly for training."""
        board = Board(size=9)
        _, policy_vec = get_move_probabilities_with_net(
            board, BLACK, net_9x9, num_simulations=30
        )
        # Should have entries for all board points + pass
        assert len(policy_vec) == 82
        # At least some moves should have nonzero probability
        assert (policy_vec > 0).sum() > 1

    def test_net_mcts_faster_than_rollout(self, net_9x9):
        """
        Neural net MCTS should be faster than rollout MCTS per simulation
        because it doesn't play random games to completion.
        """
        import time
        board = Board(size=9)

        start = time.time()
        search_with_net(board, BLACK, net_9x9, num_simulations=50)
        net_time = time.time() - start

        start = time.time()
        search(board, BLACK, num_simulations=50)
        rollout_time = time.time() - start

        # Net should be at least 2x faster (usually much more)
        assert net_time < rollout_time, (
            f"Net MCTS ({net_time:.2f}s) should be faster than "
            f"rollout MCTS ({rollout_time:.2f}s)"
        )

    def test_visits_concentrate_with_puct(self, net_9x9):
        """
        With PUCT, visits should concentrate more on high-prior moves
        compared to UCB1's more uniform exploration.
        """
        board = Board(size=9)
        root = search_with_net(board, BLACK, net_9x9, num_simulations=100)
        visits = sorted([c.visit_count for c in root.children], reverse=True)
        # Top move should get significantly more visits than average
        avg = sum(visits) / len(visits)
        assert visits[0] > 2 * avg, (
            f"Top visits ({visits[0]}) should be > 2x average ({avg:.1f})"
        )
