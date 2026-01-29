# Training Metrics & Losses (openpi)

This document explains what the current training scripts log, and what those values mean.

## 1) What is logged today

### JAX trainer (`scripts/train.py`)
- `loss`: mean of the model’s `compute_loss(...)` across batch and horizon.
- `grad_norm`: global norm of raw gradients (before optimizer clipping).
- `param_norm`: global norm of a subset of kernel parameters (bias/scale/pos/input embeddings excluded).
- `camera_views`: logged once at step 0 from the first batch for a sanity check.

### PyTorch trainer (`scripts/train_pytorch.py`)
- `loss`: mean of per-element loss from the model’s forward pass.
- `learning_rate`: current LR (cosine schedule with warmup).
- `grad_norm`: global norm of grads as returned by `clip_grad_norm_` (pre-clip norm).
- `time_per_step`: average wall-clock time per step over the log interval.
- `checkpoint_step`: logged when a checkpoint is saved.
- `camera_views`: logged once at step 0 from the first batch for a sanity check.

Note: There is no eval loop or eval metrics in the current trainers.

## 2) Loss definitions by model

### π0 / π0.5 (JAX)
Flow-matching loss on actions:
1) Sample noise `ε ~ N(0, I)` and time `t ~ Beta(1.5, 1)` in `(0, 1)`.
2) Build a noisy action `x_t = t·ε + (1 − t)·a`.
3) Target velocity `u_t = ε − a`.
4) Model predicts `v_t`.
5) Loss = mean squared error over action dimensions: `MSE(v_t, u_t)`.

The trainer averages the loss across batch and horizon.

### π0 / π0.5 (PyTorch)
Same flow-matching loss as JAX, implemented in `PI0Pytorch.forward`, returning per-element MSE with
`reduction="none"`, then the trainer averages it.

### π0-FAST (JAX only)
Autoregressive token loss:
- The model predicts the next token.
- Cross-entropy is computed only where `token_loss_mask` is true.
- `token_loss_mask` is produced by the FAST tokenizer (loss is applied only to postfix tokens).

PyTorch training does not support π0-FAST.

## 3) What is not logged by default
- No eval loss, no rollout metrics.
- No per-dimension action error.
- No policy success metrics.

