# Training Curves: Good vs Problematic Trends (openpi)

This document describes what to look for in training logs based on current instrumentation.

## 1) Metrics available today
- JAX: `loss`, `grad_norm`, `param_norm`
- PyTorch: `loss`, `learning_rate`, `grad_norm`, `time_per_step`
- Both: one-time `camera_views` sanity check

Note: There are no eval metrics unless you add them.

## 2) Generally good signals (based on current logs)
- `loss` decreases over time and then plateaus.
- `grad_norm` is finite and relatively stable (no NaNs/Infs).
- `param_norm` is finite and stable (JAX only).
- `learning_rate` follows the configured schedule (PyTorch only).
- `time_per_step` is roughly stable (PyTorch only).
- `camera_views` look correctly aligned and normalized.

## 3) Potentially problematic signals
- `loss` is NaN/Inf or diverges upward.
- `grad_norm` becomes NaN/Inf or shows repeated huge spikes.
- `param_norm` explodes or collapses to zero (JAX only).
- `time_per_step` steadily increases (PyTorch only), suggesting data-loader or memory issues.
- `camera_views` look corrupted, badly normalized, or mismatched with actions.

## 4) What’s missing (needs your guidance)
- Expected loss ranges for your specific dataset/task.
- Acceptable grad/param norm ranges.
- Task success or rollout quality metrics.
- Eval loss or per-task validation curves.

If you want concrete “good/bad” thresholds, we need to define them from prior runs or desired targets.

