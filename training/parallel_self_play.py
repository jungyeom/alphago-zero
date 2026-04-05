"""
Parallel self-play: run N games simultaneously, batching GPU evaluations.

Instead of playing games one at a time (each with its own serial MCTS),
we run N games in lockstep:
  1. For all active games, run MCTS with batched neural net calls
  2. Play the selected moves
  3. When a game ends, record it and start a new one
  4. Continue until we've completed enough games

This keeps the GPU busy with large batch sizes instead of idle between
individual forward passes.

When the C++ multi-threaded MCTS is available (--use-cpp --search-threads N),
each game uses N C++ worker threads with virtual loss + GPU batch queue,
replacing the Python MCTS entirely.
"""

import ctypes
import gc
import queue as queue_mod
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

# Configure glibc malloc to reduce heap fragmentation from C++ MCTS threads.
# - malloc_trim: force return of freed heap pages to OS
# - M_MMAP_THRESHOLD: use mmap for allocations >= 32KB (mmap is instantly returned on free)
# - M_ARENA_MAX: limit per-thread arenas to prevent unbounded growth
try:
    _libc = ctypes.CDLL("libc.so.6")
    _libc.malloc_trim.argtypes = [ctypes.c_int]
    _libc.malloc_trim.restype = ctypes.c_int
    _libc.mallopt.argtypes = [ctypes.c_int, ctypes.c_int]
    _libc.mallopt.restype = ctypes.c_int
    _M_MMAP_THRESHOLD = -3
    _M_ARENA_MAX = -8
    _libc.mallopt(_M_MMAP_THRESHOLD, 32768)   # mmap threshold: 32KB
    _libc.mallopt(_M_ARENA_MAX, 2)             # max 2 arenas
    def _malloc_trim():
        _libc.malloc_trim(0)
except (OSError, AttributeError):
    def _malloc_trim():
        pass

from go_engine.board import BLACK, WHITE, OPPONENT
from go_engine.game import Game
from model.features import board_to_features, board_to_features_cpp
from mcts.batch_search import ParallelMCTS
from training.config import TrainingConfig

# Try to import C++ multi-threaded MCTS
try:
    from mcts.cpp_search import search_parallel_cpp, _make_batch_eval_fn
    import alphago_core as _ac
    _HAS_CPP_PARALLEL = True
except (ImportError, RuntimeError):
    _HAS_CPP_PARALLEL = False


def generate_parallel_self_play(
    net: torch.nn.Module,
    config: TrainingConfig,
    device: torch.device,
    num_games: int | None = None,
    num_parallel: int = 16,
    verbose: bool = True,
) -> list[dict]:
    """
    Generate self-play games with batched inference.

    Runs `num_parallel` games simultaneously. When a game finishes,
    a new one starts in its slot. Continues until `num_games` total
    games are completed.

    Args:
        net: Current neural network
        config: Training configuration
        device: torch device
        num_games: Total games to generate
        num_parallel: Number of simultaneous games
        verbose: Print progress

    Returns:
        List of completed game data dicts
    """
    if num_games is None:
        num_games = config.games_per_iteration

    net.eval()

    # State for each parallel slot
    use_cpp = getattr(config, 'use_cpp', False)
    games: list[Game] = [Game(size=config.board_size, komi=6.5, use_cpp=use_cpp) for _ in range(num_parallel)]
    move_counts: list[int] = [0] * num_parallel
    game_data: list[dict] = [
        {"features": [], "policies": [], "colors": []} for _ in range(num_parallel)
    ]

    num_threads = getattr(config, 'num_search_threads', 1)
    use_cpp_parallel = use_cpp and _HAS_CPP_PARALLEL and num_threads > 1

    if not use_cpp_parallel:
        mcts = ParallelMCTS(
            net=net,
            num_games=num_parallel,
            board_size=config.board_size,
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_weight=config.dirichlet_weight,
            device=device,
        )
        mcts.games = games

    # Pre-create a pool of C++ MCTS search instances for concurrent game execution.
    # All instances share a single evaluator via SharedEvaluator, so all worker
    # threads from all games submit to one batch queue → large GPU batches.
    if use_cpp_parallel:
        cpp_config = _ac.MCTSConfig()
        cpp_config.num_simulations = config.num_simulations
        cpp_config.c_puct = config.c_puct
        cpp_config.dirichlet_alpha = config.dirichlet_alpha
        cpp_config.dirichlet_weight = config.dirichlet_weight
        cpp_config.num_threads = num_threads
        cpp_config.min_batch_size = getattr(config, 'min_batch_size', 4)
        cpp_config.max_batch_size = getattr(config, 'max_batch_size', 16)

        num_concurrent = getattr(config, 'num_concurrent_games', min(4, num_parallel))
        _mcts_pool_q: queue_mod.Queue = queue_mod.Queue()
        for _ in range(num_concurrent):
            _mcts_pool_q.put(_ac.MCTSSearch(cpp_config))
        cpp_batch_eval_fn = _make_batch_eval_fn(net, config.board_size, device)
        _game_executor = ThreadPoolExecutor(max_workers=num_concurrent)

        # Shared evaluator: one batch queue + one evaluator thread for ALL games.
        # This replaces per-game evaluators, giving the GPU larger batches.
        _has_shared_eval = hasattr(_ac, 'SharedEvaluator')
        if _has_shared_eval:
            _shared_eval = _ac.SharedEvaluator(
                getattr(config, 'min_batch_size', 4),
                getattr(config, 'max_batch_size', 16),
                100,  # timeout_us
            )

    # Use C++ feature extraction when available (avoids massive temp allocations)
    _extract_features = board_to_features_cpp if use_cpp else board_to_features

    completed_games: list[dict] = []
    games_since_cleanup = 0
    start_time = time.time()

    while len(completed_games) < num_games:
        # Find active (non-finished) game slots
        active_indices = [
            i for i in range(num_parallel)
            if not games[i].is_over and move_counts[i] < config.max_game_moves
        ]

        if not active_indices:
            break

        # Determine temperature for each game
        temperatures = []
        for i in active_indices:
            if move_counts[i] < config.temp_threshold:
                temperatures.append(1.0)
            else:
                temperatures.append(config.temp_final)

        # Record features for all active games
        for i in active_indices:
            color = games[i].current_player
            features = _extract_features(games[i].board, color)
            game_data[i]["features"].append(features)
            game_data[i]["colors"].append(color)

        # MCTS search for all active games
        move_results: dict[int, tuple] = {}

        if use_cpp_parallel:
            # C++ multi-threaded MCTS: run games concurrently using pool of
            # MCTSSearch instances. All share a single evaluator so worker
            # threads from ALL games feed into one batch queue → large GPU batches.
            def _run_search(args):
                game_idx, temp = args
                mcts_inst = _mcts_pool_q.get()  # borrow from pool
                try:
                    board = games[game_idx].board
                    cboard = board._board if hasattr(board, '_board') else board
                    if _has_shared_eval:
                        result = mcts_inst.search_parallel_shared(
                            cboard, games[game_idx].current_player,
                            _shared_eval, temp,
                        )
                    else:
                        result = mcts_inst.search_parallel(
                            cboard, games[game_idx].current_player,
                            cpp_batch_eval_fn, temp,
                        )
                    move = None if result.best_move.is_pass() else (result.best_move.row, result.best_move.col)
                    policy_vec = np.array(result.policy_vec, dtype=np.float32)
                    return game_idx, (move, policy_vec)
                finally:
                    _mcts_pool_q.put(mcts_inst)  # return to pool

            # Start/stop the shared evaluator around each batch of searches
            if _has_shared_eval:
                _shared_eval.start(cpp_batch_eval_fn)

            search_args = [
                (idx, temp)
                for idx, temp in zip(active_indices, temperatures)
            ]
            for game_idx, result in _game_executor.map(_run_search, search_args):
                move_results[game_idx] = result

            if _has_shared_eval:
                _shared_eval.stop()
        else:
            # Python batched MCTS: batch leaf evals across all games
            temp_groups: dict[float, list[int]] = {}
            for idx, temp in zip(active_indices, temperatures):
                temp_groups.setdefault(temp, []).append(idx)

            for temp, indices in temp_groups.items():
                results = mcts.search_batch(indices, temperature=temp)
                for idx, result in zip(indices, results):
                    move_results[idx] = result

        # Play moves and check for game end
        for i in active_indices:
            move, policy_vec = move_results[i]
            game_data[i]["policies"].append(policy_vec)

            games[i].play(move)
            move_counts[i] += 1

        # Check for completed games
        for i in range(num_parallel):
            game = games[i]
            is_done = game.is_over or move_counts[i] >= config.max_game_moves

            if is_done and game_data[i]["features"]:
                # Force end if needed
                if not game.is_over:
                    game.play(None)
                    if not game.is_over:
                        game.play(None)

                result = game.result()
                game_result = 1.0 if result["winner"] == "black" else -1.0

                completed_games.append({
                    "features": game_data[i]["features"],
                    "policies": game_data[i]["policies"],
                    "colors": game_data[i]["colors"],
                    "result": game_result,
                    "num_moves": move_counts[i],
                })

                games_since_cleanup += 1

                if verbose and len(completed_games) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = len(completed_games) / elapsed
                    print(f"  Completed {len(completed_games)}/{num_games} games "
                          f"({rate:.1f} games/min)")

                # Reset slot for a new game (if we need more games)
                if len(completed_games) < num_games:
                    games[i] = Game(size=config.board_size, komi=6.5, use_cpp=use_cpp)
                    if not use_cpp_parallel:
                        mcts.games[i] = games[i]
                    move_counts[i] = 0
                    game_data[i] = {"features": [], "policies": [], "colors": []}
                else:
                    # Mark slot as done
                    game_data[i] = {"features": [], "policies": [], "colors": []}

        # Periodic cleanup to prevent memory accumulation during self-play.
        # malloc_trim forces glibc to return freed C++ memory (MCTS tree nodes,
        # Board copies with position_history_ sets) back to the OS.
        if games_since_cleanup >= 10:
            gc.collect()
            _malloc_trim()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            games_since_cleanup = 0

    elapsed = time.time() - start_time

    if verbose:
        avg_moves = sum(g["num_moves"] for g in completed_games) / max(len(completed_games), 1)
        black_wins = sum(1 for g in completed_games if g["result"] > 0)
        print(f"  Generated {len(completed_games)} games in {elapsed:.1f}s "
              f"({len(completed_games)/elapsed*60:.1f} games/min)")
        print(f"  Avg {avg_moves:.0f} moves, black won {black_wins}/{len(completed_games)}")

    return completed_games[:num_games]
