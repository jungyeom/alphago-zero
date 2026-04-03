# AlphaZero Go

An AlphaZero-style Go AI built from scratch in Python/PyTorch with C++ acceleration. Learns to play Go through pure self-play reinforcement learning — no human game data needed.

The system combines a neural network with Monte Carlo Tree Search (MCTS). The network evaluates board positions and suggests moves; MCTS uses the network to search ahead and find stronger moves. The network then trains on the MCTS-improved policy, creating a virtuous cycle that progressively produces stronger play.

```
Neural net predicts move probabilities + position value
         |
MCTS uses the net to search 200+ moves ahead
         |
Self-play games generate training data
         |
Network trains on MCTS-improved policy
         |
Repeat -> stronger play each iteration
```

## Project structure

```
alphago-zero/
├── go_engine/              # Go rules: board state, captures, ko, scoring
│   ├── board.py            # Core board with Zobrist hashing, Chinese scoring
│   ├── cpp_board.py        # Python wrapper around C++ board
│   ├── game.py             # Game manager (turns, pass, resign)
│   └── random_play.py      # Stress test (1,100 games, 0 errors)
├── mcts/                   # Monte Carlo Tree Search
│   ├── node.py             # Tree node with UCB1 and PUCT scoring
│   ├── search.py           # MCTS with random rollouts + neural net
│   ├── batch_search.py     # Batched MCTS for parallel self-play
│   ├── cpp_search.py       # Python interface to C++ single/multi-threaded MCTS
│   └── benchmark.py        # MCTS vs random agent arena
├── model/                  # Neural network
│   ├── network.py          # ResNet with policy + value heads
│   └── features.py         # Board -> 8-channel tensor + symmetry augmentation
├── training/               # Self-play training loop
│   ├── config.py           # All hyperparameters
│   ├── self_play.py        # Generate games via MCTS + net
│   ├── parallel_self_play.py # Batched multi-game self-play
│   ├── trainer.py          # Training loop with checkpointing
│   ├── replay_buffer.py    # Position storage + sampling
│   └── logger.py           # TensorBoard logging
├── evaluation/             # Model comparison + play interface
│   ├── arena.py            # Pit two models against each other
│   └── server.py           # FastAPI backend for web UI
├── web/                    # React + TypeScript frontend
│   └── src/
│       ├── App.tsx         # Game setup + controls
│       ├── Board.tsx       # Interactive SVG Go board
│       └── api.ts          # API client
├── src/                    # C++ acceleration (pybind11)
│   ├── board.h / board.cpp           # Go board engine
│   ├── mcts_node.h                   # Tree node with PUCT + virtual loss
│   ├── mcts.h / mcts.cpp             # Single-threaded + batched MCTS
│   ├── parallel_mcts.h / .cpp        # Multi-threaded MCTS with virtual loss
│   ├── batch_queue.h                 # Promise/future GPU batch queue
│   ├── features.h / features.cpp     # Feature extraction + symmetry
│   └── bindings.cpp                  # pybind11 Python bindings
├── docs/               # Plain-English explanations of every component
├── CMakeLists.txt      # C++ build config
├── build_cpp.sh        # Build script for C++ module
├── train_local.py      # Local CPU training script (5x5, ~15 min)
└── pyproject.toml      # Managed by uv
```

## Quick start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- CMake 3.16+ (for C++ acceleration)
- Node.js 18+ (for the web UI, optional)

### Install

```bash
git clone https://github.com/jungyeom/alphago-zero.git
cd alphago-zero
uv sync
```

### Build C++ acceleration

The C++ module provides the Go board engine and multi-threaded MCTS. Build it once:

```bash
./build_cpp.sh
```

Verify it works:

```bash
uv run python -c "import alphago_core; print('OK')"
```

### Run tests

```bash
# All tests
uv run pytest

# C++ multi-threaded MCTS tests specifically
uv run pytest mcts/tests/test_parallel_mcts.py -v
```

---

## End-to-end workflow: local machine (CPU)

For testing the full pipeline on your laptop. Uses a 5x5 board with a tiny network.

### Step 1: Train on 5x5

```bash
uv run python train_local.py
```

This runs 30 iterations of self-play + training (~15 minutes on CPU). You'll see the loss decreasing each iteration. The trained model is saved to `checkpoints/model_final.pt`.

### Step 2: Play against it

```bash
# Terminal 1: backend
uv run python -m evaluation.server

# Terminal 2: frontend
cd web && npm install && npm run dev
```

Open `http://localhost:5173`, select 5x5 board size, and play.

### Step 3: Train a larger board locally (optional)

For a 9x9 model on CPU with multi-threaded C++ MCTS:

```bash
uv run python -m training.trainer \
  --board-size 9 \
  --num-iterations 30 \
  --games-per-iter 20 \
  --simulations 100 \
  --search-threads 4 \
  --use-cpp \
  --device cpu
```

The `--search-threads 4` flag enables multi-threaded MCTS, which gives ~2x speedup on CPU by overlapping C++ tree operations with neural net evaluation.

---

## End-to-end workflow: GPU instance

For real 9x9 or 13x13 training with multi-threaded MCTS and GPU acceleration.

### Step 1: Create a GPU instance

We recommend [RunPod](https://www.runpod.io):

1. Create an account and add credit ($10-25 to start)
2. Click **Pods** > **+ Deploy**
3. Select a GPU and the **RunPod PyTorch 2.x** template
4. Set **Container Disk** to 20GB, **Volume Disk** to 5GB
5. Click **Deploy On-Demand** (or **Spot** for lower cost with possible interruption)

| GPU | VRAM | On-demand | Spot | Best for |
|-----|------|-----------|------|----------|
| **RTX 4090** | 24GB | $0.34/hr | $0.20/hr | Best speed-per-dollar |
| RTX 3090 | 24GB | $0.22/hr | $0.11/hr | Cheaper, slightly slower |
| RTX 4070 Ti | 12GB | $0.19/hr | $0.10/hr | Budget option |

### Step 2: Set up the environment

Once the pod is running, click **Connect** and open **Web Terminal** (or SSH).

```bash
# Clone and checkout the branch with multi-threaded C++ MCTS
git clone https://github.com/jungyeom/alphago-zero.git
cd alphago-zero
git checkout cplus-multi-mcts

# Install uv and project dependencies
pip install uv
uv sync

# Build C++ acceleration module
./build_cpp.sh

# Verify everything works
uv run python -c "import alphago_core; print('OK')"
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name())"
```

### Step 3: Run a quick smoke test

Before a long training run, verify the full pipeline works:

```bash
uv run python -m training.trainer \
  --board-size 9 \
  --num-iterations 3 \
  --games-per-iter 5 \
  --simulations 50 \
  --search-threads 4 \
  --parallel-games 16 \
  --use-cpp \
  --device cuda
```

This should complete in under a minute. If it finishes without errors, you're ready for a real training run.

### Step 4: Start training

Run inside `tmux` so it survives SSH disconnection:

```bash
tmux new -s train
```

**9x9 training** (~4-8 hours on RTX 4090, ~$1-2 spot):

```bash
uv run python -m training.trainer \
  --board-size 9 \
  --num-iterations 150 \
  --games-per-iter 80 \
  --simulations 150 \
  --search-threads 4 \
  --parallel-games 128 \
  --use-cpp \
  --device cuda 2>&1 | tee training.log
```

**13x13 training** (~15-25 hours on RTX 4090, ~$3-5 spot):

```bash
uv run python -m training.trainer \
  --board-size 13 \
  --num-iterations 100 \
  --games-per-iter 50 \
  --simulations 150 \
  --search-threads 4 \
  --parallel-games 128 \
  --use-cpp \
  --device cuda 2>&1 | tee training.log
```

**13x13 stronger** (~30-50 hours on RTX 4090, ~$6-10 spot):

```bash
uv run python -m training.trainer \
  --board-size 13 \
  --num-iterations 200 \
  --games-per-iter 100 \
  --simulations 200 \
  --search-threads 8 \
  --parallel-games 128 \
  --use-cpp \
  --device cuda 2>&1 | tee training.log
```

Detach tmux: `Ctrl+B` then `D`. Reattach later: `tmux attach -t train`.

#### Key flags explained

| Flag | Purpose | Recommended |
|------|---------|-------------|
| `--search-threads N` | C++ worker threads per MCTS search. Higher = faster per-move search via virtual loss + GPU batching. | 4-8 on GPU |
| `--parallel-games N` | Simultaneous self-play games with batched inference. Higher = better GPU utilization. | 64-128 on GPU |
| `--use-cpp` | Use C++ board engine + MCTS instead of pure Python. | Always on GPU |
| `--simulations N` | MCTS iterations per move. More = stronger play, slower training. | 100-200 |

### Step 5: Monitor training

In a second tmux pane (`Ctrl+B` then `%`):

```bash
# Watch live progress
tail -f training.log | grep "Epoch\|Self-play\|Iteration"

# Or launch TensorBoard
uv run tensorboard --logdir runs/ --bind_all
```

For TensorBoard, forward port 6006 from your laptop:
```bash
ssh -L 6006:localhost:6006 root@<pod-ip> -p <port>
```
Then open `http://localhost:6006`.

What healthy training looks like:
- **Policy loss**: decreasing from ~3.2 to ~1.5 over 100 iterations
- **Value loss**: decreasing from ~1.0 to ~0.3, then stabilizing
- **Game length**: increasing from ~60 to ~200 as the model learns

### Step 6: Resume if interrupted

Checkpoints are saved every 5 iterations (~5MB each):

```bash
uv run python -m training.trainer \
  --resume checkpoints/model_iter_0050.pt \
  --search-threads 4 \
  --parallel-games 128 \
  --use-cpp \
  --device cuda
```

### Step 7: Download your model and play

From your laptop:

```bash
# Download (RunPod shows SSH details in the Connect tab)
scp -P <port> root@<pod-ip>:~/alphago-zero/checkpoints/model_final.pt ./checkpoints/
```

**Stop the pod** to stop billing.

Then play against it locally (no GPU needed):

```bash
# Terminal 1
uv run python -m evaluation.server

# Terminal 2
cd web && npm run dev
```

Open `http://localhost:5173`, select the board size that matches your model, and play. CPU inference takes 1-5 seconds per AI move.

---

## Comparing models

Use the arena to verify that training is making progress:

```bash
uv run python -m evaluation.arena \
  checkpoints/model_iter_0050.pt \
  checkpoints/model_iter_0100.pt \
  -n 20 --simulations 400
```

The newer model should win >55% of games if training is working.

---

## Compute budget estimates

| Board | Config | Est. time (RTX 4090) | Est. cost (spot) |
|-------|--------|---------------------|-----------------|
| 5x5 | 30 iters, 10 games, 50 sims | ~15 min | free (CPU) |
| 9x9 | 150 iters, 80 games, 150 sims, 4 threads | ~4-8 hrs | $1-2 |
| 13x13 | 100 iters, 50 games, 150 sims, 4 threads | ~15-25 hrs | $3-5 |
| 13x13 | 200 iters, 100 games, 200 sims, 8 threads | ~30-50 hrs | $6-10 |

---

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

### Multi-threaded C++ MCTS

- **N worker threads** explore the same MCTS tree concurrently using **virtual loss** to diversify exploration
- **1 evaluator thread** collects leaf positions into batches and calls the GPU for inference
- **Batch queue** with promise/future pattern decouples workers from the evaluator
- **GIL management**: workers run in pure C++ (no GIL), evaluator acquires GIL only during the Python neural net callback
- ~2.4x speedup on CPU (4 threads), higher on GPU due to better batch utilization

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
