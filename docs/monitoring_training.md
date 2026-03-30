# Monitoring Training Progress

During training, the system logs metrics to both TensorBoard (visual dashboard) and CSV (raw data). This guide explains how to use them and what to look for.

---

## 1. Starting the Dashboard

While training is running (or after it finishes), open a separate terminal and run:

```bash
cd alphago-zero
tensorboard --logdir runs/
```

Then open `http://localhost:6006` in your browser.

If training is on a remote GPU instance, forward the port via SSH:

```bash
# From your laptop
ssh -L 6006:localhost:6006 root@<pod-ip> -p <port>

# Then on the remote instance
cd alphago-zero
tensorboard --logdir runs/
```

Now open `http://localhost:6006` on your laptop — the dashboard connects through the SSH tunnel.

---

## 2. What Each Chart Means

### Loss Charts (most important)

These tell you whether the network is learning.

**loss/policy** — How well the network predicts MCTS's improved move distribution.
- Starts high (~3.0-3.5 for 13x13, which is roughly log(170) = uniform over all moves)
- Should decrease steadily
- A well-trained model reaches ~1.0-1.5
- If it plateaus early: try more MCTS simulations or lower learning rate

**loss/value** — How well the network predicts who wins from a given position.
- Starts near 1.0 (random guess)
- Should decrease to ~0.2-0.5 and stabilize
- Reaching 0.0 would mean perfect prediction (unrealistic)
- If it stays above 0.8: the network isn't learning game outcomes, try more games per iteration

**loss/total** — Sum of policy + value loss. The overall training signal.

### Self-Play Charts

These show how the games themselves change as the model improves.

**self_play/avg_game_length** — Average moves per game.
- Early training: games are short (~50-100 moves) because the model plays randomly and passes early
- As training progresses: games get longer (~150-250 moves) because the model fills the board more deliberately
- If games stay very short: the model might be learning to pass too early

**self_play/black_win_rate** — Fraction of self-play games won by black.
- Should hover around 0.45-0.55 (balanced by komi)
- If heavily skewed (>0.7 or <0.3): the network may have a color bias — usually resolves with more training

**self_play/games_per_minute** — Throughput indicator.
- On GPU with parallel self-play, expect 2-10 games/minute depending on board size and MCTS sims
- If this drops suddenly: GPU may be throttling or memory pressure

### Buffer Charts

**buffer/size** — How many positions are in the replay buffer.
- Grows until it hits the cap (50,000 by default)
- Once capped, oldest positions are replaced by new ones

### Timing Charts

**time/self_play** vs **time/training** — Where each iteration's time is spent.
- Self-play is typically 80-90% of the time
- If training time dominates: batch size may be too large or buffer too big

---

## 3. Healthy Training Progression

Here's what a successful training run looks like over 100 iterations:

```
Iteration    Policy Loss    Value Loss    Avg Game Length
    1          3.20           0.95            60
   10          2.80           0.65           100
   25          2.40           0.45           140
   50          2.00           0.35           180
   75          1.70           0.30           200
  100          1.50           0.28           210
```

The key signals:
- Policy loss drops fastest in the first 25-50 iterations
- Value loss converges earlier than policy loss
- Game length increases as the model learns to play more complete games

---

## 4. Troubleshooting

### Policy loss is flat (not decreasing)

| Possible cause | Fix |
|---------------|-----|
| Learning rate too low | Increase `learning_rate` (try 0.02) |
| Too few MCTS simulations | Increase `num_simulations` (try 300+) |
| Network too small | Increase `num_filters` or `num_blocks` |

### Policy loss spikes up

| Possible cause | Fix |
|---------------|-----|
| Learning rate too high | Decrease `learning_rate` (try 0.005) |
| Batch size too small | Increase `batch_size` |
| Replay buffer too small | Increase `replay_buffer_size` |

### Value loss stuck above 0.5

| Possible cause | Fix |
|---------------|-----|
| Too few games per iteration | Increase `games_per_iteration` |
| Games too short | Increase `max_game_moves` or check temperature schedule |
| Network too small for board size | More blocks/filters |

### All games won by one color

| Possible cause | Fix |
|---------------|-----|
| Komi imbalance | Verify komi is 6.5 |
| Bug in value perspective | Check value backup signs in MCTS |
| Early training noise | Wait 10+ iterations, usually resolves |

---

## 5. CSV Fallback

If TensorBoard isn't available, the raw metrics are saved to `runs/*/metrics.csv`:

```bash
# View latest metrics
cat runs/b13_n6_f128/metrics.csv | tail -20

# Plot with Python
python -c "
import csv, matplotlib.pyplot as plt

iters, losses = [], []
with open('runs/b13_n6_f128/metrics.csv') as f:
    for row in csv.reader(f):
        if row[2] == 'policy_loss':
            iters.append(int(row[1]))
            losses.append(float(row[3]))

plt.plot(iters, losses)
plt.xlabel('Iteration')
plt.ylabel('Policy Loss')
plt.title('Training Progress')
plt.savefig('training_curve.png')
print('Saved to training_curve.png')
"
```

---

## 6. Comparing Training Runs

TensorBoard can overlay multiple runs. If you train with different hyperparameters, each run gets its own subdirectory under `runs/`:

```
runs/
├── b13_n6_f128/    # default config
├── b13_n8_f256/    # larger network
└── b9_n6_f128/     # smaller board
```

Running `tensorboard --logdir runs/` shows all of them overlaid so you can compare which configuration learns faster.
