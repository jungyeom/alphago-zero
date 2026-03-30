"""
Training metrics logger — TensorBoard + CSV.

Logs training metrics to both TensorBoard (visual dashboard) and
a CSV file (easy to parse, survives if TensorBoard isn't available).

Usage:
    logger = TrainingLogger("runs/experiment_1")
    logger.log_training(iteration=1, policy_loss=3.2, value_loss=0.15)
    logger.log_self_play(iteration=1, num_games=100, avg_moves=200)
    logger.close()

View dashboard:
    tensorboard --logdir runs/
"""

import os
import csv
import time
from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    """
    Dual logger: TensorBoard for visual dashboards + CSV for raw data.
    """

    def __init__(self, log_dir: str = "runs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=log_dir)

        # CSV — append mode so we don't lose data on restart
        self.csv_path = os.path.join(log_dir, "metrics.csv")
        file_exists = os.path.exists(self.csv_path)
        self.csv_file = open(self.csv_path, "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        if not file_exists:
            self.csv_writer.writerow([
                "timestamp", "iteration", "metric", "value",
            ])
            self.csv_file.flush()

        self.start_time = time.time()

    def _write_csv(self, iteration: int, metric: str, value: float) -> None:
        elapsed = round(time.time() - self.start_time, 1)
        self.csv_writer.writerow([elapsed, iteration, metric, value])
        self.csv_file.flush()

    def log_training(
        self,
        iteration: int,
        policy_loss: float,
        value_loss: float,
        total_loss: float,
        learning_rate: float,
    ) -> None:
        """Log training losses after each iteration."""
        self.writer.add_scalar("loss/policy", policy_loss, iteration)
        self.writer.add_scalar("loss/value", value_loss, iteration)
        self.writer.add_scalar("loss/total", total_loss, iteration)
        self.writer.add_scalar("training/learning_rate", learning_rate, iteration)

        self._write_csv(iteration, "policy_loss", policy_loss)
        self._write_csv(iteration, "value_loss", value_loss)
        self._write_csv(iteration, "total_loss", total_loss)
        self._write_csv(iteration, "learning_rate", learning_rate)

    def log_self_play(
        self,
        iteration: int,
        num_games: int,
        avg_moves: float,
        black_win_rate: float,
        self_play_time: float,
    ) -> None:
        """Log self-play statistics."""
        self.writer.add_scalar("self_play/avg_game_length", avg_moves, iteration)
        self.writer.add_scalar("self_play/black_win_rate", black_win_rate, iteration)
        self.writer.add_scalar("self_play/games_per_minute", num_games / (self_play_time / 60), iteration)
        self.writer.add_scalar("self_play/time_seconds", self_play_time, iteration)

        self._write_csv(iteration, "avg_game_length", avg_moves)
        self._write_csv(iteration, "black_win_rate", black_win_rate)

    def log_buffer(self, iteration: int, buffer_size: int, positions_added: int) -> None:
        """Log replay buffer stats."""
        self.writer.add_scalar("buffer/size", buffer_size, iteration)
        self.writer.add_scalar("buffer/positions_added", positions_added, iteration)

    def log_eval(
        self,
        iteration: int,
        win_rate: float,
        avg_margin: float,
    ) -> None:
        """Log evaluation arena results."""
        self.writer.add_scalar("eval/win_rate_vs_previous", win_rate, iteration)
        self.writer.add_scalar("eval/avg_margin", avg_margin, iteration)

        self._write_csv(iteration, "eval_win_rate", win_rate)
        self._write_csv(iteration, "eval_margin", avg_margin)

    def log_iteration_time(self, iteration: int, total_time: float, sp_time: float, train_time: float) -> None:
        """Log timing breakdown."""
        self.writer.add_scalar("time/iteration_total", total_time, iteration)
        self.writer.add_scalar("time/self_play", sp_time, iteration)
        self.writer.add_scalar("time/training", train_time, iteration)

    def close(self) -> None:
        self.writer.close()
        self.csv_file.close()
