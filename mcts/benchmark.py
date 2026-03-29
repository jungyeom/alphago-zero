"""
Benchmark MCTS agent vs random agent.

Plays games where one side uses MCTS to choose moves and the other
plays randomly. The MCTS agent should win significantly more often.
"""

import random
import time
from go_engine.game import Game
from go_engine.board import BLACK, WHITE
from mcts.search import get_best_move


def play_mcts_vs_random(
    size: int = 9,
    mcts_color: int = BLACK,
    num_simulations: int = 50,
    max_moves: int = 300,
    verbose: bool = False,
) -> dict:
    """
    Play one game: MCTS agent vs random agent.

    Args:
        size: Board size
        mcts_color: Which color the MCTS agent plays
        num_simulations: MCTS simulations per move
        max_moves: Safety limit

    Returns:
        dict with game result
    """
    game = Game(size=size)
    move_count = 0

    while not game.is_over and move_count < max_moves:
        if game.current_player == mcts_color:
            move = get_best_move(game.board, game.current_player, num_simulations)
        else:
            legal = game.legal_moves()
            move = random.choice(legal)

        if verbose and game.current_player == mcts_color and move is not None:
            print(f"  MCTS plays: {move}")

        game.play(move)
        move_count += 1

    if not game.is_over:
        game.play(None)
        if not game.is_over:
            game.play(None)

    result = game.result()
    mcts_color_name = "black" if mcts_color == BLACK else "white"
    mcts_won = result["winner"] == mcts_color_name

    return {
        "moves": move_count,
        "winner": result["winner"],
        "mcts_won": mcts_won,
        "margin": result["margin"],
    }


def benchmark(
    num_games: int = 50,
    size: int = 9,
    num_simulations: int = 50,
    max_moves: int = 300,
) -> dict:
    """
    Run a benchmark: MCTS vs random over many games.
    Alternates MCTS between black and white for fairness.
    """
    results = []
    start_time = time.time()

    for i in range(num_games):
        mcts_color = BLACK if i % 2 == 0 else WHITE
        color_name = "black" if mcts_color == BLACK else "white"

        game_start = time.time()
        result = play_mcts_vs_random(
            size=size,
            mcts_color=mcts_color,
            num_simulations=num_simulations,
            max_moves=max_moves,
        )
        game_time = time.time() - game_start
        result["game_time"] = round(game_time, 1)
        results.append(result)

        status = "WON" if result["mcts_won"] else "LOST"
        print(f"  Game {i + 1}/{num_games}: MCTS ({color_name}) {status} "
              f"by {result['margin']:.0f} pts [{game_time:.1f}s]")

    elapsed = time.time() - start_time

    mcts_wins = sum(1 for r in results if r["mcts_won"])
    mcts_as_black_wins = sum(1 for i, r in enumerate(results) if i % 2 == 0 and r["mcts_won"])
    mcts_as_white_wins = sum(1 for i, r in enumerate(results) if i % 2 == 1 and r["mcts_won"])
    games_as_black = num_games // 2 + num_games % 2
    games_as_white = num_games // 2

    stats = {
        "num_games": num_games,
        "num_simulations": num_simulations,
        "board_size": size,
        "elapsed_seconds": round(elapsed, 1),
        "mcts_wins": mcts_wins,
        "mcts_win_rate": round(mcts_wins / num_games, 3),
        "mcts_as_black_wins": mcts_as_black_wins,
        "mcts_as_black_rate": round(mcts_as_black_wins / games_as_black, 3) if games_as_black else 0,
        "mcts_as_white_wins": mcts_as_white_wins,
        "mcts_as_white_rate": round(mcts_as_white_wins / games_as_white, 3) if games_as_white else 0,
        "avg_margin": round(sum(r["margin"] for r in results) / num_games, 1),
        "avg_moves": round(sum(r["moves"] for r in results) / num_games, 1),
    }

    return stats


def print_report(stats: dict) -> None:
    """Pretty-print benchmark results."""
    print("\n" + "=" * 55)
    print("  MCTS vs RANDOM BENCHMARK")
    print("=" * 55)
    print(f"\n  Config")
    print(f"    Board size:        {stats['board_size']}x{stats['board_size']}")
    print(f"    MCTS simulations:  {stats['num_simulations']}")
    print(f"    Games played:      {stats['num_games']}")
    print(f"    Total time:        {stats['elapsed_seconds']}s")

    print(f"\n  Results")
    print(f"    MCTS win rate:     {stats['mcts_wins']}/{stats['num_games']} "
          f"({stats['mcts_win_rate']:.1%})")
    print(f"    As black:          {stats['mcts_as_black_wins']} wins "
          f"({stats['mcts_as_black_rate']:.1%})")
    print(f"    As white:          {stats['mcts_as_white_wins']} wins "
          f"({stats['mcts_as_white_rate']:.1%})")
    print(f"    Avg margin:        {stats['avg_margin']} points")
    print(f"    Avg game length:   {stats['avg_moves']} moves")

    print(f"\n  Assessment:")
    if stats["mcts_win_rate"] >= 0.9:
        print("    MCTS is dominating random play — strong signal.")
    elif stats["mcts_win_rate"] >= 0.7:
        print("    MCTS is clearly better than random — working correctly.")
    elif stats["mcts_win_rate"] >= 0.55:
        print("    MCTS is only slightly better — may need more simulations.")
    else:
        print("    MCTS is not beating random — something may be wrong.")

    print("\n" + "=" * 55)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MCTS vs Random benchmark")
    parser.add_argument("-n", "--num-games", type=int, default=20, help="Number of games")
    parser.add_argument("-s", "--size", type=int, default=9, help="Board size")
    parser.add_argument("--sims", type=int, default=50, help="MCTS simulations per move")
    args = parser.parse_args()

    print(f"Benchmarking MCTS ({args.sims} sims) vs Random on {args.size}x{args.size}...")
    stats = benchmark(
        num_games=args.num_games,
        size=args.size,
        num_simulations=args.sims,
    )
    print_report(stats)
