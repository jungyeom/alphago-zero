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
"""

import time
import numpy as np
import torch

from go_engine.board import BLACK, WHITE, OPPONENT
from go_engine.game import Game
from model.features import board_to_features
from mcts.batch_search import ParallelMCTS
from training.config import TrainingConfig


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
    games: list[Game] = [Game(size=config.board_size) for _ in range(num_parallel)]
    move_counts: list[int] = [0] * num_parallel
    game_data: list[dict] = [
        {"features": [], "policies": [], "colors": []} for _ in range(num_parallel)
    ]

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

        # Batch MCTS search across all active games
        # Use the most common temperature (they're usually the same)
        # For simplicity, process each temperature group separately
        temp_groups: dict[float, list[int]] = {}
        for idx, temp in zip(active_indices, temperatures):
            temp_groups.setdefault(temp, []).append(idx)

        move_results: dict[int, tuple] = {}
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
                    games[i] = Game(size=config.board_size)
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
