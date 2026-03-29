"""
Evaluation arena: pit two models against each other.

Used to verify that training is making progress — a newer model
should beat an older one consistently.
"""

import time
import torch
import numpy as np

from go_engine.board import BLACK, WHITE, OPPONENT
from go_engine.game import Game
from mcts.search import get_best_move_with_net
from training.config import TrainingConfig
from training.trainer import load_checkpoint


def play_match(
    net_black: torch.nn.Module,
    net_white: torch.nn.Module,
    board_size: int = 13,
    num_simulations: int = 400,
    max_moves: int = 500,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Play one game between two models.

    Args:
        net_black: Network playing black
        net_white: Network playing white
        board_size: Board size
        num_simulations: MCTS sims per move
        max_moves: Safety limit
        device: torch device

    Returns:
        dict with game result
    """
    game = Game(size=board_size)
    nets = {BLACK: net_black, WHITE: net_white}
    move_count = 0

    while not game.is_over and move_count < max_moves:
        net = nets[game.current_player]
        move = get_best_move_with_net(
            game.board, game.current_player, net,
            num_simulations=num_simulations, device=device,
        )
        game.play(move)
        move_count += 1

    if not game.is_over:
        game.play(None)
        if not game.is_over:
            game.play(None)

    result = game.result()
    return {
        "winner": result["winner"],
        "margin": result["margin"],
        "moves": move_count,
        "reason": result["reason"],
    }


def arena(
    net_a: torch.nn.Module,
    net_b: torch.nn.Module,
    num_games: int = 20,
    board_size: int = 13,
    num_simulations: int = 400,
    device: torch.device = torch.device("cpu"),
    label_a: str = "Model A",
    label_b: str = "Model B",
) -> dict:
    """
    Play a series of games between two models, alternating colors.

    Args:
        net_a, net_b: The two networks
        num_games: Total games to play (split evenly between color assignments)
        board_size: Board size
        num_simulations: MCTS sims per move
        device: torch device
        label_a, label_b: Display names

    Returns:
        dict with aggregate results
    """
    results = []
    start = time.time()

    for i in range(num_games):
        # Alternate who plays black
        if i % 2 == 0:
            black_net, white_net = net_a, net_b
            a_color = "black"
        else:
            black_net, white_net = net_b, net_a
            a_color = "white"

        game_start = time.time()
        result = play_match(
            black_net, white_net,
            board_size=board_size,
            num_simulations=num_simulations,
            device=device,
        )
        game_time = time.time() - game_start

        a_won = result["winner"] == a_color
        result["a_won"] = a_won
        result["a_color"] = a_color
        results.append(result)

        status = f"{label_a} WON" if a_won else f"{label_b} WON"
        color_info = f"{label_a}={'B' if a_color == 'black' else 'W'}"
        print(f"  Game {i+1}/{num_games}: {status} ({color_info}, "
              f"{result['margin']:.0f} pts, {result['moves']} moves) [{game_time:.1f}s]")

    elapsed = time.time() - start
    a_wins = sum(1 for r in results if r["a_won"])
    a_as_black = sum(1 for i, r in enumerate(results) if i % 2 == 0 and r["a_won"])
    a_as_white = sum(1 for i, r in enumerate(results) if i % 2 == 1 and r["a_won"])
    games_as_black = (num_games + 1) // 2
    games_as_white = num_games // 2

    stats = {
        "num_games": num_games,
        "a_wins": a_wins,
        "b_wins": num_games - a_wins,
        "a_win_rate": a_wins / num_games,
        "a_as_black_wins": a_as_black,
        "a_as_white_wins": a_as_white,
        "games_as_black": games_as_black,
        "games_as_white": games_as_white,
        "avg_margin": np.mean([r["margin"] for r in results]),
        "avg_moves": np.mean([r["moves"] for r in results]),
        "elapsed_seconds": round(elapsed, 1),
        "label_a": label_a,
        "label_b": label_b,
    }

    return stats


def print_arena_report(stats: dict) -> None:
    """Pretty-print arena results."""
    a = stats["label_a"]
    b = stats["label_b"]

    print(f"\n{'='*55}")
    print(f"  ARENA: {a} vs {b}")
    print(f"{'='*55}")
    print(f"\n  {a} wins: {stats['a_wins']}/{stats['num_games']} "
          f"({stats['a_win_rate']:.1%})")
    print(f"    As black: {stats['a_as_black_wins']}/{stats['games_as_black']}")
    print(f"    As white: {stats['a_as_white_wins']}/{stats['games_as_white']}")
    print(f"  {b} wins: {stats['b_wins']}/{stats['num_games']} "
          f"({1 - stats['a_win_rate']:.1%})")
    print(f"\n  Avg margin: {stats['avg_margin']:.1f} points")
    print(f"  Avg game length: {stats['avg_moves']:.0f} moves")
    print(f"  Total time: {stats['elapsed_seconds']}s")

    print(f"\n  Verdict: ", end="")
    if stats["a_win_rate"] >= 0.55:
        print(f"{a} is stronger.")
    elif stats["a_win_rate"] <= 0.45:
        print(f"{b} is stronger.")
    else:
        print("No clear winner — roughly equal strength.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model evaluation arena")
    parser.add_argument("model_a", help="Path to first model checkpoint")
    parser.add_argument("model_b", help="Path to second model checkpoint")
    parser.add_argument("-n", "--num-games", type=int, default=20)
    parser.add_argument("-s", "--simulations", type=int, default=400)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    from training.trainer import get_device

    config = TrainingConfig(device=args.device)
    device = get_device(config)

    net_a, ckpt_a = load_checkpoint(args.model_a, device)
    net_b, ckpt_b = load_checkpoint(args.model_b, device)

    iter_a = ckpt_a.get("iteration", "?")
    iter_b = ckpt_b.get("iteration", "?")

    print(f"Model A: iter {iter_a} | Model B: iter {iter_b}")
    print(f"Device: {device} | Simulations: {args.simulations}")

    stats = arena(
        net_a, net_b,
        num_games=args.num_games,
        board_size=ckpt_a["config"]["board_size"],
        num_simulations=args.simulations,
        device=device,
        label_a=f"Iter {iter_a}",
        label_b=f"Iter {iter_b}",
    )
    print_arena_report(stats)
