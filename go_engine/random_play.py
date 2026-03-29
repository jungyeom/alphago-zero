"""
Random self-play stress test for the Go rules engine.

Plays many games of random legal moves and collects statistics to verify:
- All games terminate
- Scoring works without errors
- No crashes from edge cases (ko, suicide, captures)
- Game lengths and scores look reasonable
"""

import random
import time
from go_engine.game import Game
from go_engine.board import BLACK, WHITE


def play_random_game(size: int = 13, max_moves: int = 500, seed: int | None = None) -> dict:
    """
    Play a single game of random legal moves.

    Args:
        size: Board size
        max_moves: Safety limit to prevent infinite games
        seed: Random seed for reproducibility

    Returns:
        dict with game statistics
    """
    if seed is not None:
        random.seed(seed)

    game = Game(size=size)
    move_count = 0

    while not game.is_over and move_count < max_moves:
        legal = game.legal_moves()
        move = random.choice(legal)
        game.play(move)
        move_count += 1

    if not game.is_over:
        # Hit max_moves — force end with two passes
        game.play(None)
        if not game.is_over:
            game.play(None)

    result = game.result()
    return {
        "moves": move_count,
        "winner": result["winner"],
        "black_score": result["black"],
        "white_score": result["white"],
        "margin": result["margin"],
        "reason": result["reason"],
        "captures_black": game.board.captured[BLACK],
        "captures_white": game.board.captured[WHITE],
    }


def stress_test(
    num_games: int = 500,
    size: int = 13,
    max_moves: int = 500,
    verbose: bool = True,
) -> dict:
    """
    Play many random games and report aggregate statistics.

    Returns:
        dict with aggregate stats
    """
    results = []
    errors = []
    start_time = time.time()

    for i in range(num_games):
        try:
            result = play_random_game(size=size, max_moves=max_moves, seed=i)
            results.append(result)
        except Exception as e:
            errors.append({"game": i, "error": str(e), "type": type(e).__name__})

        if verbose and (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  {i + 1}/{num_games} games ({elapsed:.1f}s)")

    elapsed = time.time() - start_time

    # Aggregate stats
    if not results:
        return {"errors": errors, "elapsed": elapsed}

    move_counts = [r["moves"] for r in results]
    margins = [r["margin"] for r in results]
    black_wins = sum(1 for r in results if r["winner"] == "black")
    white_wins = sum(1 for r in results if r["winner"] == "white")
    draws = sum(1 for r in results if r["winner"] == "draw")
    forced_ends = sum(1 for r in results if r["moves"] >= max_moves)
    total_captures_b = sum(r["captures_black"] for r in results)
    total_captures_w = sum(r["captures_white"] for r in results)

    stats = {
        "num_games": len(results),
        "num_errors": len(errors),
        "errors": errors,
        "elapsed_seconds": round(elapsed, 2),
        "games_per_second": round(len(results) / elapsed, 2),
        "black_wins": black_wins,
        "white_wins": white_wins,
        "draws": draws,
        "black_win_rate": round(black_wins / len(results), 3),
        "forced_ends": forced_ends,
        "moves_avg": round(sum(move_counts) / len(move_counts), 1),
        "moves_min": min(move_counts),
        "moves_max": max(move_counts),
        "margin_avg": round(sum(margins) / len(margins), 1),
        "avg_captures_black": round(total_captures_b / len(results), 1),
        "avg_captures_white": round(total_captures_w / len(results), 1),
    }

    return stats


def print_report(stats: dict) -> None:
    """Pretty-print the stress test results."""
    print("\n" + "=" * 55)
    print("  RANDOM SELF-PLAY STRESS TEST REPORT")
    print("=" * 55)

    if stats.get("num_errors", 0) > 0:
        print(f"\n  *** {stats['num_errors']} ERRORS DETECTED ***")
        for err in stats["errors"][:10]:
            print(f"    Game {err['game']}: {err['type']}: {err['error']}")
    else:
        print(f"\n  No errors across {stats['num_games']} games")

    print(f"\n  Performance")
    print(f"    Total time:        {stats['elapsed_seconds']}s")
    print(f"    Games/second:      {stats['games_per_second']}")

    print(f"\n  Game Length")
    print(f"    Average moves:     {stats['moves_avg']}")
    print(f"    Range:             {stats['moves_min']} - {stats['moves_max']}")
    print(f"    Forced ends:       {stats['forced_ends']} (hit max_moves limit)")

    print(f"\n  Win Rates")
    print(f"    Black:             {stats['black_wins']} ({stats['black_win_rate']:.1%})")
    print(f"    White:             {stats['white_wins']} ({1 - stats['black_win_rate']:.1%})")
    print(f"    Draws:             {stats['draws']}")
    print(f"    Avg margin:        {stats['margin_avg']} points")

    print(f"\n  Captures")
    print(f"    Avg by black:      {stats['avg_captures_black']}")
    print(f"    Avg by white:      {stats['avg_captures_white']}")

    print("\n" + "=" * 55)

    # Sanity checks
    print("\n  Sanity Checks:")
    checks = [
        ("No errors", stats["num_errors"] == 0),
        ("All games terminated", stats["forced_ends"] == 0),
        ("Both sides win some", stats["black_wins"] > 0 and stats["white_wins"] > 0),
        ("Win rate roughly balanced (30-70%)", 0.3 <= stats["black_win_rate"] <= 0.7),
        ("Average game > 50 moves", stats["moves_avg"] > 50),
        ("Captures happening", stats["avg_captures_black"] > 0),
    ]
    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"    [{status}] {name}")

    print("\n" + ("  All checks passed!" if all_pass else "  *** SOME CHECKS FAILED ***"))
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Random self-play stress test")
    parser.add_argument("-n", "--num-games", type=int, default=500, help="Number of games")
    parser.add_argument("-s", "--size", type=int, default=13, help="Board size")
    parser.add_argument("--max-moves", type=int, default=500, help="Max moves per game")
    args = parser.parse_args()

    print(f"Running {args.num_games} random games on {args.size}x{args.size}...")
    stats = stress_test(
        num_games=args.num_games,
        size=args.size,
        max_moves=args.max_moves,
    )
    print_report(stats)
