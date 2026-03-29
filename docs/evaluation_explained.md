# How the Evaluation Arena Works

The arena answers a simple question: **is the new model better than the old one?**

---

## 1. Why We Need This

Training loss going down doesn't guarantee the model plays better Go. The loss could decrease while the model develops blind spots or overfit patterns. The only true test is playing games.

The arena loads two checkpoints, has them play a series of games against each other, and reports who wins more.

---

## 2. How It Works

1. Load two model checkpoints (e.g., iteration 50 vs iteration 100)
2. Play N games, alternating which model gets black and which gets white
3. Each move is selected by MCTS with the model's network (more simulations than training for stronger play)
4. Report win rates, margins, and a verdict

Color alternation is critical — without it, komi advantage would bias results.

---

## 3. Usage

### Compare two checkpoints
```bash
uv run python -m evaluation.arena checkpoints/model_iter_0050.pt checkpoints/model_iter_0100.pt \
  -n 20 --simulations 400
```

### What to look for

| Signal | Meaning |
|--------|---------|
| New model wins >55% | Training is improving — keep going |
| Win rates ~50% | Model has plateaued — try different hyperparameters |
| New model wins <45% | Training went wrong — might need to revert |
| Increasing game length over iterations | Models are playing more carefully — good sign |

---

## 4. Integration with Training

During training, the arena runs automatically every `eval_every_n_iterations` iterations (default: 10). It compares the current model against the best known model. If the new model wins convincingly (>55%), it becomes the new best.

This "gatekeeper" prevents the training from regressing — a concept called **self-play with model selection** in the AlphaZero paper.

---

## 5. Limitations

- **Small sample sizes are noisy**: 20 games can't distinguish a 52% model from a 48% model. For serious evaluation, use 100+ games.
- **Simulation count matters**: models compared at 20 sims might rank differently at 400 sims. Always evaluate at the same sim count.
- **Early models all look similar**: untrained networks play essentially randomly, so the first ~10 iterations of arena results are not meaningful.
