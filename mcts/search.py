"""
Monte Carlo Tree Search (MCTS).

This module implements the core MCTS algorithm in two flavors:
1. UCT (Upper Confidence bounds applied to Trees) — uses random rollouts
   to evaluate leaf positions. No neural network needed.
2. AlphaZero-style — uses a neural network for both move priors and
   position evaluation. Implemented in Step 5.

The search works by repeatedly:
  SELECT → EXPAND → EVALUATE → BACKUP
"""

import random
import numpy as np
import torch
from go_engine.board import Board, BLACK, WHITE, OPPONENT
from go_engine.game import Game
from mcts.node import Node


def _random_rollout(board: Board, color_to_play: int, max_moves: int = 200) -> float:
    """
    Play random moves until the game ends and return the result.

    Optimized: instead of generating all legal moves each turn, we sample
    random empty intersections and check legality for just that one point.
    After enough failures, we pass. Two consecutive passes end the game.

    Args:
        board: Current board state (will be copied, not modified)
        color_to_play: Whose turn it is
        max_moves: Safety limit

    Returns:
        +1 if black wins, -1 if white wins
    """
    import numpy as np

    sim_board = board.copy()
    color = color_to_play
    consecutive_passes = 0
    max_attempts = board.size * board.size  # give up and pass after this many tries

    for _ in range(max_moves):
        if consecutive_passes >= 2:
            break

        # Find empty points
        empty = np.argwhere(sim_board.grid == 0)
        if len(empty) == 0:
            consecutive_passes += 1
            color = OPPONENT[color]
            continue

        # Shuffle and try random empty points
        np.random.shuffle(empty)
        played = False
        attempts = min(len(empty), max_attempts)
        for idx in range(attempts):
            r, c = int(empty[idx][0]), int(empty[idx][1])
            if sim_board.is_legal(color, (r, c)):
                sim_board.play(color, (r, c))
                color = OPPONENT[color]
                consecutive_passes = 0
                played = True
                break

        if not played:
            # No legal move found among sampled points — pass
            sim_board.play(color, None)
            color = OPPONENT[color]
            consecutive_passes += 1

    score = sim_board.score()
    return 1.0 if score["winner"] == "black" else -1.0


def _select(node: Node) -> Node:
    """
    Walk down the tree, always picking the child with the highest UCB1 score,
    until we reach a node that hasn't been expanded yet.
    """
    current = node
    while current.is_expanded:
        current = current.best_child(use_puct=False)
    return current


def _expand(node: Node, board: Board, color_to_play: int) -> None:
    """
    Expand a leaf node by creating child nodes for all legal moves.

    Each child gets a uniform prior (1/num_moves) since we don't have
    a neural network yet.
    """
    legal = board.legal_moves(color_to_play)
    num_moves = len(legal)
    uniform_prior = 1.0 / num_moves if num_moves > 0 else 0.0

    for move in legal:
        child = Node(
            move=move,
            parent=node,
            color=color_to_play,
            prior=uniform_prior,
        )
        node.children.append(child)


def _backup(node: Node, value: float) -> None:
    """
    Propagate the evaluation value back up the tree.

    At each node, we store the value from the perspective of the player
    who made the move. Since value is from black's perspective (+1 = black wins),
    we negate it at each level to flip between perspectives.
    """
    current = node
    # value is from black's perspective.
    # At each node, we store it relative to the player who played that move.
    while current is not None:
        current.visit_count += 1
        # If the node's color is BLACK, the value is as-is.
        # If the node's color is WHITE, negate it.
        if current.color == BLACK:
            current.total_value += value
        elif current.color == WHITE:
            current.total_value -= value
        else:
            # Root node (color=0), just store raw value
            current.total_value += value
        current = current.parent


def _apply_move(board: Board, color: int, move: tuple[int, int] | None) -> Board:
    """Apply a move to a board copy and return the new board."""
    new_board = board.copy()
    new_board.play(color, move)
    return new_board


def search(
    board: Board,
    color_to_play: int,
    num_simulations: int = 100,
    c: float = 1.4,
) -> Node:
    """
    Run MCTS from the given position and return the root node.

    The best move can be extracted via root.most_visited_child().move

    Args:
        board: Current board state
        color_to_play: BLACK or WHITE
        num_simulations: Number of MCTS iterations
        c: Exploration constant for UCB1

    Returns:
        Root node of the search tree (children contain move statistics)
    """
    root = Node(move=None, parent=None, color=0)
    _expand(root, board, color_to_play)

    for _ in range(num_simulations):
        # 1. SELECT — walk down the tree to a leaf
        leaf = _select(root)

        # 2. Reconstruct the board state at this leaf by replaying moves
        node_path = []
        current = leaf
        while current is not root:
            node_path.append(current)
            current = current.parent

        # Replay moves from root to leaf
        sim_board = board.copy()
        sim_color = color_to_play
        for node in reversed(node_path):
            if node.move is not None or node.move is None:
                sim_board.play(sim_color, node.move)
                sim_color = OPPONENT[sim_color]

        # 3. EXPAND the leaf (if the game isn't over at this position)
        game_over = False
        # Check if game ended (two consecutive passes)
        if len(node_path) >= 2:
            last_two = node_path[:2]  # most recent two moves
            if last_two[0].move is None and last_two[1].move is None:
                game_over = True

        if not game_over and not leaf.is_expanded:
            _expand(leaf, sim_board, sim_color)

        # 4. EVALUATE — random rollout from this position
        if game_over:
            # Score directly
            score = sim_board.score()
            value = 1.0 if score["winner"] == "black" else -1.0
        else:
            value = _random_rollout(sim_board, sim_color)

        # 5. BACKUP — propagate value up the tree
        _backup(leaf, value)

    return root


def get_best_move(
    board: Board,
    color_to_play: int,
    num_simulations: int = 100,
) -> tuple[int, int] | None:
    """
    Convenience function: run MCTS and return the best move.

    Args:
        board: Current board state
        color_to_play: BLACK or WHITE
        num_simulations: Number of MCTS iterations

    Returns:
        Best move as (row, col) or None for pass
    """
    root = search(board, color_to_play, num_simulations)
    best = root.most_visited_child()
    return best.move


def get_move_probabilities(
    board: Board,
    color_to_play: int,
    num_simulations: int = 100,
    temperature: float = 1.0,
) -> list[tuple[tuple[int, int] | None, float]]:
    """
    Run MCTS and return move probabilities based on visit counts.

    With temperature=1.0, probabilities are proportional to visit counts.
    With temperature→0, this becomes argmax (deterministic).

    This is what we'll later use to generate training data:
    the MCTS policy is the training target for the neural network.

    Returns:
        List of (move, probability) pairs
    """
    root = search(board, color_to_play, num_simulations)

    if not root.children:
        return [(None, 1.0)]

    if temperature < 1e-8:
        # Deterministic — all weight on most visited
        best = root.most_visited_child()
        return [(child.move, 1.0 if child is best else 0.0) for child in root.children]

    # Apply temperature to visit counts
    visits = [child.visit_count ** (1.0 / temperature) for child in root.children]
    total = sum(visits)

    if total == 0:
        # All unvisited — uniform
        uniform = 1.0 / len(root.children)
        return [(child.move, uniform) for child in root.children]

    return [(child.move, v / total) for child, v in zip(root.children, visits)]


# ═══════════════════════════════════════════════════════════════════
# AlphaZero-style MCTS with Neural Network (Step 5)
# ═══════════════════════════════════════════════════════════════════


def _expand_with_net(
    node: Node,
    board: Board,
    color_to_play: int,
    net: torch.nn.Module,
    device: torch.device,
) -> float:
    """
    Expand a leaf node using the neural network.

    The network provides:
      - policy priors for each child (guides search toward promising moves)
      - a value estimate (replaces the random rollout)

    Args:
        node: Leaf node to expand
        board: Board state at this node
        color_to_play: Whose turn it is
        net: The AlphaZero network
        device: torch device (cpu/cuda)

    Returns:
        Value estimate from the network's perspective (from black's POV)
    """
    from model.features import board_to_tensor

    # Get network prediction
    x = board_to_tensor(board, color_to_play).to(device)
    policy, value = net.predict(x)
    policy = policy[0].cpu().numpy()  # shape: (num_moves,)
    value_scalar = value[0, 0].item()  # scalar in [-1, 1]

    # The network outputs value from the current player's perspective.
    # Convert to black's perspective for consistent backup.
    if color_to_play == WHITE:
        value_scalar = -value_scalar

    # Get legal moves and mask the policy
    legal = board.legal_moves(color_to_play)
    size = board.size

    # Build a mask for legal moves
    legal_mask = np.zeros(size * size + 1, dtype=np.float32)
    for move in legal:
        if move is None:
            legal_mask[-1] = 1.0  # pass
        else:
            r, c = move
            legal_mask[r * size + c] = 1.0

    # Zero out illegal moves and renormalize
    masked_policy = policy * legal_mask
    policy_sum = masked_policy.sum()
    if policy_sum > 1e-8:
        masked_policy /= policy_sum
    else:
        # All legal moves got ~0 probability — use uniform over legal moves
        masked_policy = legal_mask / legal_mask.sum()

    # Create child nodes with network priors
    for move in legal:
        if move is None:
            prior = float(masked_policy[-1])
        else:
            r, c = move
            prior = float(masked_policy[r * size + c])

        child = Node(
            move=move,
            parent=node,
            color=color_to_play,
            prior=prior,
        )
        node.children.append(child)

    return value_scalar


def search_with_net(
    board: Board,
    color_to_play: int,
    net: torch.nn.Module,
    num_simulations: int = 200,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.1,
    dirichlet_weight: float = 0.25,
    device: torch.device | None = None,
) -> Node:
    """
    Run AlphaZero-style MCTS using a neural network.

    Key differences from random-rollout MCTS:
    - Uses PUCT formula instead of UCB1 (incorporates network priors)
    - Network evaluates leaf positions directly (no rollout needed)
    - Dirichlet noise added to root priors for exploration

    Args:
        board: Current board state
        color_to_play: BLACK or WHITE
        net: AlphaZero network
        num_simulations: Number of MCTS iterations
        c_puct: Exploration constant for PUCT
        dirichlet_alpha: Dirichlet noise parameter (~0.1 for 13x13)
        dirichlet_weight: How much noise to mix in (0.25 = 25% noise)
        device: torch device

    Returns:
        Root node of the search tree
    """
    if device is None:
        device = next(net.parameters()).device

    root = Node(move=None, parent=None, color=0)

    # Expand root with network
    _expand_with_net(root, board, color_to_play, net, device)

    # Add Dirichlet noise to root priors for exploration
    if root.children and dirichlet_alpha > 0:
        noise = np.random.dirichlet([dirichlet_alpha] * len(root.children))
        for child, n in zip(root.children, noise):
            child.prior = (1 - dirichlet_weight) * child.prior + dirichlet_weight * n

    for _ in range(num_simulations):
        # 1. SELECT — walk down tree using PUCT
        current = root
        while current.is_expanded:
            current = current.best_child(use_puct=True, c=c_puct)
        leaf = current

        # 2. Reconstruct board state at leaf
        node_path = []
        cur = leaf
        while cur is not root:
            node_path.append(cur)
            cur = cur.parent

        sim_board = board.copy()
        sim_color = color_to_play
        for node in reversed(node_path):
            sim_board.play(sim_color, node.move)
            sim_color = OPPONENT[sim_color]

        # 3. Check for game over
        game_over = False
        if len(node_path) >= 2:
            if node_path[0].move is None and node_path[1].move is None:
                game_over = True

        # 4. EVALUATE
        if game_over:
            score = sim_board.score()
            value = 1.0 if score["winner"] == "black" else -1.0
        elif not leaf.is_expanded:
            # Expand with network — this returns the value AND creates children
            value = _expand_with_net(leaf, sim_board, sim_color, net, device)
        else:
            # Already expanded (shouldn't happen often)
            value = 0.0

        # 5. BACKUP
        _backup(leaf, value)

    return root


def get_best_move_with_net(
    board: Board,
    color_to_play: int,
    net: torch.nn.Module,
    num_simulations: int = 200,
    device: torch.device | None = None,
) -> tuple[int, int] | None:
    """Run MCTS with neural net and return the best move."""
    root = search_with_net(board, color_to_play, net, num_simulations, device=device)
    return root.most_visited_child().move


def get_move_probabilities_with_net(
    board: Board,
    color_to_play: int,
    net: torch.nn.Module,
    num_simulations: int = 200,
    temperature: float = 1.0,
    device: torch.device | None = None,
) -> tuple[list[tuple[tuple[int, int] | None, float]], np.ndarray]:
    """
    Run MCTS with neural net and return move probabilities.

    Returns:
        Tuple of:
          - List of (move, probability) pairs
          - Policy vector as numpy array of shape (board_size**2 + 1,)
            suitable for training (indexed by move position)
    """
    root = search_with_net(board, color_to_play, net, num_simulations, device=device)
    size = board.size

    if not root.children:
        policy_vec = np.zeros(size * size + 1, dtype=np.float32)
        policy_vec[-1] = 1.0
        return [(None, 1.0)], policy_vec

    # Build visit count vector
    visits_vec = np.zeros(size * size + 1, dtype=np.float32)
    for child in root.children:
        if child.move is None:
            visits_vec[-1] = child.visit_count
        else:
            r, c = child.move
            visits_vec[r * size + c] = child.visit_count

    # Apply temperature
    if temperature < 1e-8:
        # Deterministic
        policy_vec = np.zeros_like(visits_vec)
        policy_vec[np.argmax(visits_vec)] = 1.0
    else:
        visits_temp = visits_vec ** (1.0 / temperature)
        total = visits_temp.sum()
        policy_vec = visits_temp / total if total > 0 else visits_vec

    # Also build the list-of-tuples format
    move_probs = []
    for child in root.children:
        if child.move is None:
            move_probs.append((None, float(policy_vec[-1])))
        else:
            r, c = child.move
            move_probs.append((child.move, float(policy_vec[r * size + c])))

    return move_probs, policy_vec
