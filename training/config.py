"""
All training hyperparameters in one place.

Change these values to tune the training pipeline.
Defaults are calibrated for 13x13 on ~$50 of GPU compute.
"""

from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    # ── Board ────────────────────────────────────────────────────────
    board_size: int = 13

    # ── Network ─────────────────────────────────────────────────────
    num_blocks: int = 6
    num_filters: int = 128

    # ── MCTS (self-play) ────────────────────────────────────────────
    num_simulations: int = 200       # MCTS sims per move during self-play
    c_puct: float = 1.5              # exploration constant for PUCT
    dirichlet_alpha: float = 0.1     # noise parameter (~10/board_size²)
    dirichlet_weight: float = 0.25   # mix ratio: 75% net + 25% noise

    # ── Temperature schedule ────────────────────────────────────────
    # Moves 0..temp_threshold use temp=1.0 (exploratory)
    # Moves after temp_threshold use temp=temp_final (exploitative)
    temp_threshold: int = 15
    temp_final: float = 0.1

    # ── Self-play ───────────────────────────────────────────────────
    games_per_iteration: int = 100   # games to generate per training iteration
    max_game_moves: int = 500        # safety cap on game length

    # ── Replay buffer ───────────────────────────────────────────────
    replay_buffer_size: int = 50_000  # max positions stored
    min_buffer_size: int = 2_000      # train only when buffer has this many

    # ── Training ────────────────────────────────────────────────────
    batch_size: int = 256
    training_epochs: int = 3         # epochs per iteration over sampled data
    learning_rate: float = 0.01
    lr_decay_steps: list[int] = field(default_factory=lambda: [100, 200])
    lr_decay_factor: float = 0.1
    weight_decay: float = 1e-4       # L2 regularization
    use_mixed_precision: bool = True  # fp16 for ~2x GPU speedup

    # ── Augmentation ────────────────────────────────────────────────
    use_symmetry_augmentation: bool = True  # 8x data via board symmetries

    # ── Checkpointing ───────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    save_every_n_iterations: int = 5
    num_iterations: int = 200        # total training iterations

    # ── Evaluation ──────────────────────────────────────────────────
    eval_games: int = 20             # games to play for model comparison
    eval_simulations: int = 400      # MCTS sims per move during eval (more than training)
    eval_every_n_iterations: int = 10

    # ── Device ──────────────────────────────────────────────────────
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
