"""
Python interface to the C++ MCTS search.

The entire MCTS loop (select, expand, backup, board reconstruction) runs in C++.
Only neural net evaluation crosses back to Python for GPU inference.
This eliminates the pybind11 per-call overhead that makes the wrapper approach slow.
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import alphago_core as ac
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

from go_engine.board import BLACK, WHITE


def _make_eval_fn(net: torch.nn.Module, board_size: int, device: torch.device):
    """
    Create a C++ NetEvalFn callback that evaluates positions using the PyTorch net.

    The C++ MCTS calls this function when it reaches a leaf node.
    It receives a C++ Board and color, and returns NetOutput (policy + value).
    """
    from model.features import NUM_FEATURES

    def eval_fn(cboard, color):
        """Single position evaluation."""
        # Extract features from C++ board
        last_r, last_c = -1, -1
        features = ac.board_to_features_cpp(cboard, color, last_r, last_c)

        # Convert to tensor and run network
        tensor = torch.from_numpy(features).unsqueeze(0).to(device)
        net.eval()
        with torch.no_grad():
            log_policy, value = net(tensor)
            policy = torch.exp(log_policy)[0].cpu().numpy()
            value_scalar = value[0, 0].item()

        out = ac.NetOutput()
        out.policy = policy.tolist()
        out.value = value_scalar
        return out

    return eval_fn


def _make_batch_eval_fn(net: torch.nn.Module, board_size: int, device: torch.device):
    """
    Create a batch evaluation callback for the C++ MCTS.

    Evaluates multiple positions in a single GPU forward pass.
    Uses a pre-allocated CPU tensor buffer to avoid repeated allocations
    that fragment the C heap.
    """
    from model.features import NUM_FEATURES

    # Pre-allocate CPU batch buffer (reused across calls to avoid malloc churn).
    # Using numpy backing array so C++ can write directly into it via board_to_features_into.
    max_batch = 32
    _np_buf = np.zeros((max_batch, NUM_FEATURES, board_size, board_size), dtype=np.float32)
    _batch_buf = torch.from_numpy(_np_buf)  # shares memory with _np_buf
    if device.type == "cuda":
        _gpu_buf = torch.zeros((max_batch, NUM_FEATURES, board_size, board_size),
                               dtype=torch.float32, device=device)
    _has_features_into = hasattr(ac, 'board_to_features_into')

    def batch_eval_fn(cboards, colors):
        """Batch position evaluation."""
        n = len(cboards)
        if n == 0:
            return []

        # Extract features directly into pre-allocated buffer (zero-copy)
        if n <= max_batch and _has_features_into:
            for i in range(n):
                ac.board_to_features_into(cboards[i], colors[i], _np_buf, i)
            batch = _batch_buf[:n].to(device, non_blocking=True)
        else:
            # Fallback for oversized batches
            tensors = []
            for i in range(n):
                features = ac.board_to_features_cpp(cboards[i], colors[i])
                tensors.append(torch.from_numpy(features).unsqueeze(0))
            batch = torch.cat(tensors, dim=0).to(device)

        net.eval()
        with torch.no_grad():
            log_policy, value = net(batch)
            policies = torch.exp(log_policy).cpu().numpy()
            values = value.cpu().numpy().flatten()

        results = []
        for i in range(n):
            out = ac.NetOutput()
            out.policy = policies[i].tolist()
            out.value = float(values[i])
            results.append(out)

        return results

    return batch_eval_fn


def search_cpp(
    board,
    color_to_play: int,
    net: torch.nn.Module,
    num_simulations: int = 200,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.1,
    dirichlet_weight: float = 0.25,
    temperature: float = 1.0,
    device: torch.device | None = None,
) -> tuple[tuple[int, int] | None, np.ndarray]:
    """
    Run MCTS entirely in C++ with neural net callback to Python.

    Args:
        board: Python Board or CppBoard
        color_to_play: BLACK or WHITE
        net: PyTorch network
        num_simulations: MCTS iterations
        c_puct: exploration constant
        dirichlet_alpha: noise parameter
        dirichlet_weight: noise mixing weight
        temperature: move selection temperature
        device: torch device

    Returns:
        (best_move, policy_vector) — same format as get_move_probabilities_with_net
    """
    if not HAS_CPP:
        raise RuntimeError("C++ module not available")

    if device is None:
        device = next(net.parameters()).device

    board_size = board.size if isinstance(board.size, int) else board.size()

    # Get or create C++ board
    if hasattr(board, '_board'):
        cboard = board._board
    else:
        # Convert Python Board to C++ Board
        cboard = ac.CBoard(board_size)
        grid = board.grid
        for r in range(board_size):
            for c in range(board_size):
                if grid[r, c] != 0:
                    cboard.play(int(grid[r, c]), ac.board_move(r, c))

    # Configure MCTS
    config = ac.MCTSConfig()
    config.num_simulations = num_simulations
    config.c_puct = c_puct
    config.dirichlet_alpha = dirichlet_alpha
    config.dirichlet_weight = dirichlet_weight

    mcts = ac.MCTSSearch(config)
    eval_fn = _make_eval_fn(net, board_size, device)

    # Run search entirely in C++
    result = mcts.search(cboard, color_to_play, eval_fn, temperature)

    # Convert result
    move = None if result.best_move.is_pass() else (result.best_move.row, result.best_move.col)
    policy_vec = np.array(result.policy_vec, dtype=np.float32)

    return move, policy_vec


def get_move_probabilities_cpp(
    board,
    color_to_play: int,
    net: torch.nn.Module,
    num_simulations: int = 200,
    temperature: float = 1.0,
    device: torch.device | None = None,
) -> tuple[list[tuple[tuple[int, int] | None, float]], np.ndarray]:
    """
    C++ MCTS that returns the same format as get_move_probabilities_with_net.
    """
    move, policy_vec = search_cpp(
        board, color_to_play, net,
        num_simulations=num_simulations,
        temperature=temperature,
        device=device,
    )

    # Build move_probs list from policy_vec
    board_size = board.size if isinstance(board.size, int) else board.size()
    move_probs = []
    for i in range(board_size * board_size):
        if policy_vec[i] > 0:
            r, c = i // board_size, i % board_size
            move_probs.append(((r, c), float(policy_vec[i])))
    # Pass
    pass_prob = float(policy_vec[-1])
    if pass_prob > 0:
        move_probs.append((None, pass_prob))

    return move_probs, policy_vec


def search_parallel_cpp(
    board,
    color_to_play: int,
    net: torch.nn.Module,
    num_simulations: int = 800,
    num_threads: int = 4,
    min_batch_size: int = 4,
    max_batch_size: int = 16,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.1,
    dirichlet_weight: float = 0.25,
    temperature: float = 1.0,
    device: torch.device | None = None,
) -> tuple[tuple[int, int] | None, np.ndarray]:
    """
    Multi-threaded C++ MCTS with virtual loss and GPU batch queue.

    Multiple C++ threads explore the same tree simultaneously.
    Leaf evaluations are batched and sent to the GPU together.
    """
    if not HAS_CPP:
        raise RuntimeError("C++ module not available")

    if device is None:
        device = next(net.parameters()).device

    board_size = board.size if isinstance(board.size, int) else board.size()

    if hasattr(board, '_board'):
        cboard = board._board
    else:
        cboard = ac.CBoard(board_size)
        grid = board.grid
        for r in range(board_size):
            for c in range(board_size):
                if grid[r, c] != 0:
                    cboard.play(int(grid[r, c]), ac.board_move(r, c))

    config = ac.MCTSConfig()
    config.num_simulations = num_simulations
    config.c_puct = c_puct
    config.dirichlet_alpha = dirichlet_alpha
    config.dirichlet_weight = dirichlet_weight
    config.num_threads = num_threads
    config.min_batch_size = min_batch_size
    config.max_batch_size = max_batch_size

    mcts = ac.MCTSSearch(config)
    batch_eval_fn = _make_batch_eval_fn(net, board_size, device)

    # This releases the GIL internally, runs parallel C++ threads,
    # evaluator thread acquires GIL for batch_eval_fn calls
    result = mcts.search_parallel(cboard, color_to_play, batch_eval_fn, temperature)

    move = None if result.best_move.is_pass() else (result.best_move.row, result.best_move.col)
    policy_vec = np.array(result.policy_vec, dtype=np.float32)

    return move, policy_vec


def get_move_probabilities_parallel_cpp(
    board,
    color_to_play: int,
    net: torch.nn.Module,
    num_simulations: int = 800,
    num_threads: int = 4,
    temperature: float = 1.0,
    device: torch.device | None = None,
    min_batch_size: int = 4,
    max_batch_size: int = 16,
) -> tuple[list[tuple[tuple[int, int] | None, float]], np.ndarray]:
    """
    Multi-threaded C++ MCTS returning same format as get_move_probabilities_with_net.
    """
    move, policy_vec = search_parallel_cpp(
        board, color_to_play, net,
        num_simulations=num_simulations,
        num_threads=num_threads,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        temperature=temperature,
        device=device,
    )

    board_size = board.size if isinstance(board.size, int) else board.size()
    move_probs = []
    for i in range(board_size * board_size):
        if policy_vec[i] > 0:
            r, c = i // board_size, i % board_size
            move_probs.append(((r, c), float(policy_vec[i])))
    pass_prob = float(policy_vec[-1])
    if pass_prob > 0:
        move_probs.append((None, pass_prob))

    return move_probs, policy_vec
