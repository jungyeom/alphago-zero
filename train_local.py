"""
Local training script for testing on CPU.

Uses a small 5x5 board with reduced parameters so you can verify
the full pipeline works before spending money on cloud GPU.

Expected runtime: ~10-15 minutes on a laptop.
After training, test it with the web UI.
"""

from training.config import TrainingConfig
from training.trainer import train


config = TrainingConfig(
    # Small board — trainable on CPU
    board_size=5,

    # Tiny network (still learns patterns on 5x5)
    num_blocks=3,
    num_filters=64,

    # Fewer MCTS sims — faster self-play
    num_simulations=50,

    # Fewer games per iteration — faster iterations
    games_per_iteration=10,

    # Training
    batch_size=64,
    training_epochs=3,
    learning_rate=0.01,
    lr_decay_steps=[15, 25],
    replay_buffer_size=10_000,
    min_buffer_size=200,

    # 30 iterations — enough to see improvement on 5x5
    num_iterations=30,
    save_every_n_iterations=10,

    # Temperature
    temp_threshold=8,  # fewer opening moves on 5x5
    temp_final=0.1,

    # CPU
    device="cpu",
    use_mixed_precision=False,

    # Eval
    eval_every_n_iterations=10,
    eval_games=10,
    eval_simulations=100,
)


if __name__ == "__main__":
    print("=" * 55)
    print("  LOCAL TRAINING TEST (5x5, CPU)")
    print("=" * 55)
    print(f"\n  Board:        {config.board_size}x{config.board_size}")
    print(f"  Network:      {config.num_blocks} blocks, {config.num_filters} filters")
    print(f"  MCTS sims:    {config.num_simulations}")
    print(f"  Games/iter:   {config.games_per_iteration}")
    print(f"  Iterations:   {config.num_iterations}")
    print(f"  Device:       {config.device}")
    print(f"\n  Estimated time: ~10-15 minutes")
    print(f"  Checkpoint dir: {config.checkpoint_dir}/")
    print()

    train(config)

    print("\n" + "=" * 55)
    print("  Training complete!")
    print(f"  Model saved to: {config.checkpoint_dir}/model_final.pt")
    print(f"\n  To play against it:")
    print(f"    Terminal 1: uv run python -m evaluation.server")
    print(f"    Terminal 2: cd web && npm run dev")
    print(f"    Then open http://localhost:5173")
    print(f"    Select 5x5 board size")
    print("=" * 55)
