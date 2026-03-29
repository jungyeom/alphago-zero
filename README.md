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

## Training on a GPU

The local test uses a 5x5 board on CPU. For real 13x13 training, you need a GPU.

### Recommended setup

| Provider | GPU | Cost | Notes |
|----------|-----|------|-------|
| RunPod | A10G (24GB) | ~$0.50/hr | Good balance of price and speed |
| Lambda | A10G | ~$0.75/hr | Simple setup |
| RunPod | L4 (24GB) | ~$0.40/hr | Slightly slower, cheaper |

### Setup on a GPU instance

```bash
# Clone and install
git clone https://github.com/jungyeom/alphago-zero.git
cd alphago-zero
pip install uv    # if uv not available
uv sync

# Verify GPU is available
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### Run training

```bash
# Full 13x13 training (default config)
uv run python -m training.trainer \
  --board-size 13 \
  --num-iterations 200 \
  --games-per-iter 100 \
  --simulations 200 \
  --device cuda

# Budget-conscious: fewer iterations, smaller simulations
uv run python -m training.trainer \
  --board-size 13 \
  --num-iterations 100 \
  --games-per-iter 50 \
  --simulations 100 \
  --device cuda

# Start with 9x9 (faster, cheaper, still interesting)
uv run python -m training.trainer \
  --board-size 9 \
  --num-iterations 150 \
  --games-per-iter 80 \
  --simulations 150 \
  --device cuda
```

### Resume from checkpoint

If your instance gets interrupted or you want to continue training:

```bash
uv run python -m training.trainer \
  --resume checkpoints/model_iter_0050.pt \
  --device cuda
```

### Compute budget estimates

| Board | Iterations | Games/iter | Sims | Est. time | Est. cost (~$1/hr) |
|-------|-----------|------------|------|-----------|-------------------|
| 5x5 | 30 | 10 | 50 | ~15 min | free (CPU) |
| 9x9 | 150 | 80 | 150 | ~15-25 hrs | $15-25 |
| 13x13 | 200 | 100 | 200 | ~30-60 hrs | $30-60 |

### Download your model

After training, copy the checkpoint back to your laptop:

```bash
scp gpu-instance:~/alphago-zero/checkpoints/model_final.pt ./checkpoints/
```

Then run the web UI locally to play against it.

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
- [Evaluation explained](docs/evaluation_explained.md) — model comparison arena
- [Web UI explained](docs/web_ui_explained.md) — frontend/backend architecture, API reference

## License

MIT
