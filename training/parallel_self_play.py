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

import time
import numpy as np
import torch

from go_engine.board import BLACK, WHITE, OPPONENT
from go_engine.game import Game
from model.features import board_to_features
from mcts.batch_search import ParallelMCTS
from training.config import TrainingConfig

# Try to import C++ multi-threaded MCTS
try:
    from mcts.cpp_search import search_parallel_cpp
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

    completed_games: list[dict] = []
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
            features = board_to_features(games[i].board, color)
            game_data[i]["features"].append(features)
            game_data[i]["colors"].append(color)

        # MCTS search for all active games
        move_results: dict[int, tuple] = {}

        if use_cpp_parallel:
            # C++ multi-threaded MCTS: each game runs with N worker threads
            min_bs = getattr(config, 'min_batch_size', 4)
            max_bs = getattr(config, 'max_batch_size', 16)
            for idx, temp in zip(active_indices, temperatures):
                move, policy_vec = search_parallel_cpp(
                    games[idx].board,
                    games[idx].current_player,
                    net,
                    num_simulations=config.num_simulations,
                    num_threads=num_threads,
                    min_batch_size=min_bs,
                    max_batch_size=max_bs,
                    c_puct=config.c_puct,
                    dirichlet_alpha=config.dirichlet_alpha,
                    dirichlet_weight=config.dirichlet_weight,
                    temperature=temp,
                    device=device,
                )
                move_results[idx] = (move, policy_vec)
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

    elapsed = time.time() - start_time

    if verbose:
        avg_moves = sum(g["num_moves"] for g in completed_games) / max(len(completed_games), 1)
        black_wins = sum(1 for g in completed_games if g["result"] > 0)
        print(f"  Generated {len(completed_games)} games in {elapsed:.1f}s "
              f"({len(completed_games)/elapsed*60:.1f} games/min)")
        print(f"  Avg {avg_moves:.0f} moves, black won {black_wins}/{len(completed_games)}")

    return completed_games[:num_games]
