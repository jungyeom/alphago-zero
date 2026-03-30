# AlphaZero Go

An AlphaZero-style Go AI built from scratch in Python/PyTorch. Learns to play Go through pure self-play reinforcement learning — no human game data needed.

The system combines a neural network with Monte Carlo Tree Search (MCTS). The network evaluates board positions and suggests moves; MCTS uses the network to search ahead and find stronger moves. The network then trains on the MCTS-improved policy, creating a virtuous cycle that progressively produces stronger play.

## How it works

```
Neural net predicts move probabilities + position value
         ↓
MCTS uses the net to search 200+ moves ahead
         ↓
Self-play games generate training data
         ↓
Network trains on MCTS-improved policy
         ↓
Repeat → stronger play each iteration
```

## Project structure

```
alphago-zero/
├── go_engine/          # Go rules: board state, captures, ko, scoring
│   ├── board.py        # Core board with Zobrist hashing, Chinese scoring
│   ├── game.py         # Game manager (turns, pass, resign)
│   └── random_play.py  # Stress test (1,100 games, 0 errors)
├── mcts/               # Monte Carlo Tree Search
│   ├── node.py         # Tree node with UCB1 and PUCT scoring
│   ├── search.py       # MCTS with random rollouts + neural net
│   └── benchmark.py    # MCTS vs random agent arena
├── model/              # Neural network
│   ├── network.py      # ResNet with policy + value heads
│   └── features.py     # Board → 8-channel tensor + symmetry augmentation
├── training/           # Self-play training loop
│   ├── config.py       # All hyperparameters
│   ├── self_play.py    # Generate games via MCTS + net
│   ├── trainer.py      # Training loop with checkpointing
│   └── replay_buffer.py
├── evaluation/         # Model comparison + play interface
│   ├── arena.py        # Pit two models against each other
│   └── server.py       # FastAPI backend for web UI
├── web/                # React + TypeScript frontend
│   └── src/
│       ├── App.tsx     # Game setup + controls
│       ├── Board.tsx   # Interactive SVG Go board
│       └── api.ts      # API client
├── docs/               # Plain-English explanations of every component
├── train_local.py      # Local CPU training script (5x5, ~15 min)
└── pyproject.toml      # Managed by uv
```

## Quick start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Node.js 18+ (for the web UI)

### Install

```bash
git clone https://github.com/jungyeom/alphago-zero.git
cd alphago-zero
uv sync
cd web && npm install && cd ..
```

### Local test training (CPU, ~15 minutes)

Train a small model on 5x5 to verify everything works:

```bash
uv run python train_local.py
```

This runs 30 iterations of self-play + training on a 5x5 board with a small network. You'll see the loss decreasing each iteration. The trained model is saved to `checkpoints/model_final.pt`.

### Play against the AI

After training, start the backend and frontend:

```bash
# Terminal 1: backend
uv run python -m evaluation.server

# Terminal 2: frontend
cd web && npm run dev
```

Open `http://localhost:5173`, select the board size that matches your trained model, and play. The server auto-detects the latest checkpoint in `checkpoints/`.

## Training on a GPU (end-to-end guide)

The local test uses a 5x5 board on CPU. For real 13x13 training, you need a GPU instance.

### Step 1: Create a GPU instance on RunPod

1. Create an account at [runpod.io](https://www.runpod.io) and add credit ($10-25 to start)
2. Click **Pods** in the left sidebar, then **+ Deploy**
3. Select **RTX 4090** (24GB) — best speed/cost for our model size
4. Choose the **RunPod PyTorch 2.x** template
5. Set **Container Disk** to 20GB, **Volume Disk** to 5GB
6. Click **Deploy On-Demand** (or **Spot** for ~$0.20/hr with possible interruption)

| GPU | VRAM | On-demand | Spot | Best for |
|-----|------|-----------|------|----------|
| **RTX 4090** | 24GB | $0.34/hr | $0.20/hr | Best speed-per-dollar |
| RTX 4070 Ti | 12GB | $0.19/hr | $0.10/hr | Budget option |
| RTX 3090 | 24GB | $0.22/hr | $0.11/hr | Cheaper, slightly slower |

### Step 2: Connect and set up

Once the pod is running, click **Connect** and open **Web Terminal** (or SSH).

```bash
# Clone the repo (use mcts-optimizer branch for C++ acceleration)
git clone https://github.com/jungyeom/alphago-zero.git
cd alphago-zero
git checkout mcts-optimizer

# Install dependencies
pip install uv
uv sync

# Build C++ acceleration (15x faster board operations)
./build_cpp.sh

# Verify GPU is available
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

### Step 3: Start training

Run training inside `tmux` or `screen` so it survives SSH disconnection:

```bash
tmux new -s train

# Recommended 13x13 config for ~$5-10 budget
uv run python -m training.trainer \
  --board-size 13 \
  --num-iterations 100 \
  --games-per-iter 50 \
  --simulations 150 \
  --use-cpp \
  --device cuda 2>&1 | tee training.log

# Detach tmux: Ctrl+B then D
# Reattach later: tmux attach -t train
```

Other training configs:

```bash
# Budget-conscious (~$3-5)
uv run python -m training.trainer \
  --board-size 13 --num-iterations 50 --games-per-iter 30 \
  --simulations 100 --use-cpp --device cuda

# Stronger model (~$10-17)
uv run python -m training.trainer \
  --board-size 13 --num-iterations 200 --games-per-iter 100 \
  --simulations 200 --use-cpp --device cuda

# 9x9 (faster, good for experimentation)
uv run python -m training.trainer \
  --board-size 9 --num-iterations 150 --games-per-iter 80 \
  --simulations 150 --use-cpp --device cuda
```

### Step 4: Monitor training

In a second terminal (or another tmux pane):

```bash
# Watch live loss values
tail -f training.log | grep "Epoch\|Self-play\|Iteration"

# Or launch TensorBoard dashboard
uv run tensorboard --logdir runs/ --bind_all
```

If using TensorBoard, forward port 6006 from your laptop:
```bash
# From your laptop
ssh -L 6006:localhost:6006 root@<pod-ip> -p <port>
```
Then open `http://localhost:6006` to see live loss curves.

What healthy training looks like:
- **Policy loss**: decreasing from ~3.2 to ~1.5 over 100 iterations
- **Value loss**: decreasing from ~1.0 to ~0.3, then stabilizing
- **Game length**: increasing from ~60 to ~200 as the model learns

See [docs/monitoring_training.md](docs/monitoring_training.md) for detailed troubleshooting.

### Step 5: Resume if interrupted

If the instance stops (spot reclaimed, SSH drops, etc.), your checkpoints are saved:

```bash
# Resume from the latest checkpoint
uv run python -m training.trainer \
  --resume checkpoints/model_iter_0050.pt \
  --use-cpp --device cuda
```

Checkpoints are saved every 5 iterations (~5MB each). At most you lose the current in-progress iteration.

### Step 6: Download your model

When training finishes, copy the checkpoint to your laptop:

```bash
# From your laptop (RunPod shows SSH details in the Connect tab)
scp -P <port> root@<pod-ip>:~/alphago-zero/checkpoints/model_final.pt ./checkpoints/
```

**Stop the pod** to stop billing.

### Step 7: Play against it locally

Back on your laptop, no GPU needed:

```bash
# Terminal 1: start the backend (auto-loads checkpoints/model_final.pt)
uv run python -m evaluation.server

# Terminal 2: start the frontend
cd web && npm run dev
```

Open `http://localhost:5173`, select 13x13, and play. CPU inference takes 1-5 seconds per AI move.

### Compute budget estimates

| Board | Config | Est. time (RTX 4090) | Est. cost (spot) |
|-------|--------|---------------------|-----------------|
| 5x5 | 30 iters, 10 games, 50 sims | ~15 min | free (CPU) |
| 9x9 | 150 iters, 80 games, 150 sims | ~8-15 hrs | $2-3 |
| 13x13 | 100 iters, 50 games, 150 sims | ~15-25 hrs | $3-5 |
| 13x13 | 200 iters, 100 games, 200 sims | ~30-50 hrs | $6-10 |

## Comparing models

Use the arena to verify that training is making progress:

```bash
uv run python -m evaluation.arena \
  checkpoints/model_iter_0050.pt \
  checkpoints/model_iter_0100.pt \
  -n 20 --simulations 400
```

The newer model should win >55% of games if training is working.

## Architecture details

### Neural network

- **ResNet** with 6 residual blocks, 128 filters (~1.2M parameters)
- **Policy head**: probability distribution over all board points + pass
- **Value head**: scalar in [-1, +1] estimating who is winning
- **Input**: 8-channel tensor (current stones, opponent stones, empty, last move, color, liberties)

### MCTS

- **PUCT** selection formula with neural network priors
- **Dirichlet noise** at root for exploration (alpha=0.1 for 13x13)
- **Temperature schedule**: exploratory for first 15 moves, then deterministic
- No random rollouts — the neural network directly evaluates leaf positions

### Training

- **Self-play** generates training data (the network plays both sides)
- **Replay buffer** stores up to 50K positions with random sampling
- **8x data augmentation** via board symmetries (rotations + reflections)
- **Loss**: cross-entropy (policy) + MSE (value) + L2 regularization
- **Optimizer**: SGD with momentum 0.9, weight decay 1e-4

## Documentation

The `docs/` folder contains plain-English explanations of every component:

- [Go engine explained](docs/go_engine_explained.md) — board representation, captures, ko, scoring
- [MCTS explained](docs/mcts_explained.md) — search algorithm, UCB1/PUCT, rollouts vs neural net
- [Neural net explained](docs/neural_net_explained.md) — ResNet architecture, features, training
- [Training explained](docs/training_explained.md) — self-play loop, replay buffer, temperature
- [Monitoring training](docs/monitoring_training.md) — TensorBoard dashboard, what to watch, troubleshooting
- [Evaluation explained](docs/evaluation_explained.md) — model comparison arena
- [Web UI explained](docs/web_ui_explained.md) — frontend/backend architecture, API reference

## License

MIT
