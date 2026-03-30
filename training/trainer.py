"""
Training loop: self-play → replay buffer → train network → repeat.

This is the core AlphaZero training pipeline. Each iteration:
  1. Generate self-play games using the current network
  2. Store positions in the replay buffer
  3. Train the network on sampled positions
  4. Periodically evaluate against previous versions
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from model.network import create_network
from training.config import TrainingConfig
from training.replay_buffer import ReplayBuffer
from training.self_play import generate_self_play_data


def get_device(config: TrainingConfig) -> torch.device:
    """Resolve the device string to a torch.device."""
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(config.device)


def train_batch(
    net: nn.Module,
    optimizer: optim.Optimizer,
    features: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    scaler: GradScaler | None = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Train on a single batch.

    Loss = cross_entropy(policy) + MSE(value)

    The network outputs log-probabilities for policy, so we use NLL loss.

    Returns:
        dict with 'policy_loss', 'value_loss', 'total_loss'
    """
    net.train()
    features = features.to(device)
    policy_targets = policy_targets.to(device)
    value_targets = value_targets.to(device)

    optimizer.zero_grad()

    use_amp = scaler is not None and device.type == "cuda"

    if use_amp:
        with autocast():
            log_policy, value = net(features)
            policy_loss = -torch.mean(torch.sum(policy_targets * log_policy, dim=1))
            value_loss = torch.mean((value - value_targets) ** 2)
            loss = policy_loss + value_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        log_policy, value = net(features)
        policy_loss = -torch.mean(torch.sum(policy_targets * log_policy, dim=1))
        value_loss = torch.mean((value - value_targets) ** 2)
        loss = policy_loss + value_loss

        loss.backward()
        optimizer.step()

    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "total_loss": loss.item(),
    }


def train_epoch(
    net: nn.Module,
    optimizer: optim.Optimizer,
    buffer: ReplayBuffer,
    config: TrainingConfig,
    scaler: GradScaler | None = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Train for one epoch: sample batches from the replay buffer.

    Returns:
        dict with average losses for the epoch
    """
    num_batches = max(1, len(buffer) // config.batch_size)
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_loss = 0.0

    for _ in range(num_batches):
        features_np, policies_np, values_np = buffer.sample(config.batch_size)

        features = torch.from_numpy(features_np)
        policy_targets = torch.from_numpy(policies_np)
        value_targets = torch.from_numpy(values_np)

        losses = train_batch(
            net, optimizer, features, policy_targets, value_targets,
            scaler=scaler, device=device,
        )

        total_policy_loss += losses["policy_loss"]
        total_value_loss += losses["value_loss"]
        total_loss += losses["total_loss"]

    return {
        "policy_loss": total_policy_loss / num_batches,
        "value_loss": total_value_loss / num_batches,
        "total_loss": total_loss / num_batches,
    }


def save_checkpoint(
    net: nn.Module,
    optimizer: optim.Optimizer,
    iteration: int,
    config: TrainingConfig,
    path: str | None = None,
) -> str:
    """Save model checkpoint. Returns the path."""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    if path is None:
        path = os.path.join(config.checkpoint_dir, f"model_iter_{iteration:04d}.pt")

    torch.save({
        "iteration": iteration,
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "board_size": config.board_size,
            "num_blocks": config.num_blocks,
            "num_filters": config.num_filters,
        },
    }, path)
    return path


def load_checkpoint(
    path: str,
    device: torch.device = torch.device("cpu"),
) -> tuple[nn.Module, dict]:
    """Load a model from checkpoint. Returns (net, checkpoint_dict)."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    net = create_network(
        board_size=cfg["board_size"],
        num_blocks=cfg["num_blocks"],
        num_filters=cfg["num_filters"],
    )
    net.load_state_dict(checkpoint["model_state_dict"])
    net.to(device)
    return net, checkpoint


def train(config: TrainingConfig | None = None, resume_from: str | None = None) -> None:
    """
    Main training loop.

    Args:
        config: Training configuration (uses defaults if None)
        resume_from: Path to checkpoint to resume from
    """
    if config is None:
        config = TrainingConfig()

    device = get_device(config)
    print(f"Using device: {device}")

    # Create or load network
    if resume_from:
        print(f"Resuming from {resume_from}")
        net, checkpoint = load_checkpoint(resume_from, device)
        start_iteration = checkpoint["iteration"] + 1
    else:
        net = create_network(
            board_size=config.board_size,
            num_blocks=config.num_blocks,
            num_filters=config.num_filters,
        )
        net.to(device)
        start_iteration = 1

    print(f"Network: {net.count_parameters():,} parameters")

    # Optimizer
    optimizer = optim.SGD(
        net.parameters(),
        lr=config.learning_rate,
        momentum=0.9,
        weight_decay=config.weight_decay,
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.lr_decay_steps,
        gamma=config.lr_decay_factor,
    )

    # Resume optimizer state if available
    if resume_from and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for _ in range(start_iteration - 1):
            scheduler.step()

    # Mixed precision
    scaler = GradScaler() if config.use_mixed_precision and device.type == "cuda" else None

    # Replay buffer
    buffer = ReplayBuffer(max_size=config.replay_buffer_size)

    # Logger (TensorBoard + CSV)
    from training.logger import TrainingLogger
    log_dir = os.path.join("runs", f"b{config.board_size}_n{config.num_blocks}_f{config.num_filters}")
    logger = TrainingLogger(log_dir=log_dir)
    print(f"Logging to: {log_dir}/")
    print(f"  View dashboard: tensorboard --logdir runs/")

    # Training loop
    for iteration in range(start_iteration, config.num_iterations + 1):
        iter_start = time.time()
        print(f"\n{'='*60}")
        print(f"  Iteration {iteration}/{config.num_iterations}  "
              f"(lr={optimizer.param_groups[0]['lr']:.6f})")
        print(f"{'='*60}")

        # 1. Generate self-play games
        print("\n  Phase 1: Self-play")
        sp_start = time.time()
        if config.use_parallel_self_play and config.num_parallel_games > 1:
            from training.parallel_self_play import generate_parallel_self_play
            print(f"  (parallel: {config.num_parallel_games} games batched)")
            games = generate_parallel_self_play(
                net, config, device,
                num_parallel=config.num_parallel_games,
            )
        else:
            games = generate_self_play_data(net, config, device)
        sp_time = time.time() - sp_start

        # 2. Add to replay buffer
        positions_added = 0
        for game_data in games:
            n = buffer.push_game(
                features_list=game_data["features"],
                policies_list=game_data["policies"],
                colors_list=game_data["colors"],
                game_result=game_data["result"],
                board_size=config.board_size,
                use_symmetry=config.use_symmetry_augmentation,
            )
            positions_added += n

        # Self-play stats
        avg_moves = sum(g["num_moves"] for g in games) / len(games)
        black_wins = sum(1 for g in games if g["result"] > 0)
        black_win_rate = black_wins / len(games)

        print(f"  Added {positions_added} positions "
              f"(buffer: {len(buffer)}/{config.replay_buffer_size})")
        print(f"  Self-play time: {sp_time:.1f}s")

        logger.log_self_play(iteration, len(games), avg_moves, black_win_rate, sp_time)
        logger.log_buffer(iteration, len(buffer), positions_added)

        # 3. Train the network
        if len(buffer) >= config.min_buffer_size:
            print(f"\n  Phase 2: Training ({config.training_epochs} epochs)")
            train_start = time.time()

            for epoch in range(config.training_epochs):
                losses = train_epoch(
                    net, optimizer, buffer, config,
                    scaler=scaler, device=device,
                )
                print(f"    Epoch {epoch + 1}: "
                      f"policy={losses['policy_loss']:.4f} "
                      f"value={losses['value_loss']:.4f} "
                      f"total={losses['total_loss']:.4f}")

            train_time = time.time() - train_start
            print(f"  Training time: {train_time:.1f}s")

            lr = optimizer.param_groups[0]["lr"]
            logger.log_training(iteration, losses["policy_loss"], losses["value_loss"], losses["total_loss"], lr)
        else:
            train_time = 0.0
            print(f"\n  Skipping training (buffer {len(buffer)} < {config.min_buffer_size})")

        # 4. Step the learning rate scheduler
        scheduler.step()

        # 5. Save checkpoint
        if iteration % config.save_every_n_iterations == 0:
            path = save_checkpoint(net, optimizer, iteration, config)
            print(f"\n  Saved checkpoint: {path}")

        iter_time = time.time() - iter_start
        print(f"\n  Iteration time: {iter_time:.1f}s")
        logger.log_iteration_time(iteration, iter_time, sp_time, train_time)

    # Save final model
    final_path = save_checkpoint(net, optimizer, config.num_iterations, config,
                                  path=os.path.join(config.checkpoint_dir, "model_final.pt"))
    logger.close()
    print(f"\nTraining complete. Final model: {final_path}")
    print(f"View training dashboard: tensorboard --logdir runs/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AlphaZero training")
    parser.add_argument("--board-size", type=int, default=13)
    parser.add_argument("--num-iterations", type=int, default=200)
    parser.add_argument("--games-per-iter", type=int, default=100)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use-cpp", action="store_true", default=False,
                        help="Use C++ board engine (run ./build_cpp.sh first)")
    parser.add_argument("--parallel-games", type=int, default=16,
                        help="Number of simultaneous self-play games (default: 16, try 64-128 on GPU)")
    parser.add_argument("--search-threads", type=int, default=1,
                        help="MCTS threads per game (default: 1, try 4-8 on GPU with --use-cpp)")
    args = parser.parse_args()

    config = TrainingConfig(
        board_size=args.board_size,
        num_iterations=args.num_iterations,
        games_per_iteration=args.games_per_iter,
        num_simulations=args.simulations,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        use_cpp=args.use_cpp,
        num_parallel_games=args.parallel_games,
        num_search_threads=args.search_threads,
    )

    train(config, resume_from=args.resume)
