"""
Self-play game generation.

Uses the current neural network + MCTS to play games against itself,
producing training data (state, MCTS policy, game outcome).
"""

import numpy as np
import torch

from go_engine.board import BLACK, WHITE, OPPONENT
from go_engine.game import Game
from model.features import board_to_features
from mcts.search import get_move_probabilities_with_net
from training.config import TrainingConfig


def play_self_play_game(
    net: torch.nn.Module,
    config: TrainingConfig,
    device: torch.device,
    verbose: bool = False,
) -> dict:
    """
    Play one self-play game and return the training data.

    The network plays both sides. MCTS generates an improved policy
    at each position, which becomes the training target.

    Returns:
        dict with:
          - features: list of feature arrays (C, H, W)
          - policies: list of policy vectors (num_moves,)
          - colors: list of colors (who was to play)
          - result: +1 if black won, -1 if white won
          - num_moves: total moves in the game
    """
    game = Game(size=config.board_size, komi=6.5)

    features_list = []
    policies_list = []
    colors_list = []

    move_count = 0

    while not game.is_over and move_count < config.max_game_moves:
        color = game.current_player

        # Temperature schedule: exploratory early, exploitative later
        if move_count < config.temp_threshold:
            temperature = 1.0
        else:
            temperature = config.temp_final

        # Record features BEFORE the move
        features = board_to_features(game.board, color)

        # Run MCTS to get improved policy
        move_probs, policy_vec = get_move_probabilities_with_net(
            game.board,
            color,
            net,
            num_simulations=config.num_simulations,
            temperature=temperature,
            device=device,
        )

        # Store training data
        features_list.append(features)
        policies_list.append(policy_vec)
        colors_list.append(color)

        # Select move from the MCTS policy
        if temperature < 1e-8:
            # Deterministic: pick the most visited
            best_move = max(move_probs, key=lambda x: x[1])[0]
        else:
            # Sample from the distribution
            moves = [m for m, _ in move_probs]
            probs = np.array([p for _, p in move_probs], dtype=np.float64)
            probs /= probs.sum()  # renormalize for fp precision
            idx = np.random.choice(len(moves), p=probs)
            best_move = moves[idx]

        game.play(best_move)
        move_count += 1

        if verbose and move_count % 20 == 0:
            turn = "B" if color == BLACK else "W"
            print(f"    Move {move_count}: {turn} plays {best_move}")

    # Force game end if needed
    if not game.is_over:
        game.play(None)
        if not game.is_over:
            game.play(None)

    result = game.result()
    game_result = 1.0 if result["winner"] == "black" else -1.0

    if verbose:
        winner = result["winner"]
        margin = result["margin"]
        print(f"    Game over: {winner} wins by {margin:.1f} ({move_count} moves)")

    return {
        "features": features_list,
        "policies": policies_list,
        "colors": colors_list,
        "result": game_result,
        "num_moves": move_count,
    }


def generate_self_play_data(
    net: torch.nn.Module,
    config: TrainingConfig,
    device: torch.device,
    num_games: int | None = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Generate multiple self-play games.

    Args:
        net: Current neural network
        config: Training configuration
        device: torch device
        num_games: Number of games (defaults to config.games_per_iteration)
        verbose: Print progress

    Returns:
        List of game data dicts (each from play_self_play_game)
    """
    if num_games is None:
        num_games = config.games_per_iteration

    net.eval()
    games = []

    for i in range(num_games):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Self-play game {i + 1}/{num_games}")

        game_data = play_self_play_game(net, config, device, verbose=False)
        games.append(game_data)

    if verbose:
        avg_moves = sum(g["num_moves"] for g in games) / len(games)
        black_wins = sum(1 for g in games if g["result"] > 0)
        print(f"  Generated {num_games} games, avg {avg_moves:.0f} moves, "
              f"black won {black_wins}/{num_games}")

    return games
