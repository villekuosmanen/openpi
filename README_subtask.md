# Pi0.5 Subtask Training Guide (LIBERO)

## Overview

Pi0.5 with subtask capability supports **two training strategies**:

| Strategy | Description |
|----------|-------------|
| **Joint Training** | Train subtask prediction, discrete action token prediction, and continuous action generation in a single run |
| **Knowledge Insulation** | Two-stage training: first finetune the VLM, then finetune the action expert ([paper](https://www.pi.website/research/knowledge_insulation)) |

---

## Broad Codebase Changes

This feature is not just a new training config. It adds a hierarchical
task->subtask->action path across the model, tokenizer, transforms, dataset
adapters, and evaluation scripts.

### Historical context

The original open-source release already included Pi0.5-related code paths and continuous action generation, but it did not include the full paper-style Pi0.5 training path described in the `π0.5` paper.

In particular, the released code did not include a complete training setup that combined all of the following in one Pi0.5 path:

- explicit subtask text prediction
- explicit FAST action-token prediction
- explicit hierarchical inference where low-level actions are conditioned on a predicted subtask

This subtask feature adds that missing training/inference plumbing. In other words, it is not just a decoding change; it changes the prompt structure, loss structure, data transforms, and runtime path needed to train and run a paper-style hierarchical Pi0.5 model.

### 1. New Pi0.5 subtask-capable model path

The subtask feature adds a dedicated Pi0.5 implementation and config:

- `src/openpi/models/pi05.py`
- `src/openpi/models/pi05_config.py`
- `src/openpi/models/gemma_05.py`
- `src/openpi/models/pi05_test.py`

These files introduce Pi0.5-specific training behavior, including separate weights for:

- subtask text generation loss
- FAST action-token loss
- continuous action flow-matching loss

### 2. Observation format extended for hierarchical token regions

The generic model input structure was extended so training can distinguish subtask-token supervision from action-token supervision:

- `src/openpi/models/model.py`

New observation fields include:

- `subtask_region_mask`
- `action_region_mask`

These masks are used by Pi0.5 training to apply different losses to different token regions of the prompt.

### 3. Tokenizer and transform pipeline extended

The original single-prompt pipeline was extended to support a hierarchical prompt layout:

- high-level task
- low-level subtask
- optional FAST action-token segment

Main files:

- `src/openpi/models/tokenizer.py`
- `src/openpi/transforms.py`

Key additions:

- `TokenizeHighLowPrompt` in `src/openpi/transforms.py`
- tokenizer support for `subtask_region_mask` and `action_region_mask` in `src/openpi/models/tokenizer.py`

### 4. Subtask-aware LIBERO dataset and policy adapters

The training data path was extended so LIBERO samples can carry both a high-level task and a low-level subtask:

- `src/openpi/policies/libero_subtask_policy.py`
- `src/openpi/training/config.py`

`LeRobotLiberoSubtaskDataConfig` in `src/openpi/training/config.py` defines the subtask-aware data mapping. It expects fields like:

- `task`
- `subtask`
- `images.agentview_rgb`
- `images.wrist_rgb_left`
- `state`
- `actions`

### 5. Training recipes added in config

The README training modes are backed by concrete configs in:

- `src/openpi/training/config.py`

Main configs added:

- `libero_pi05_subtask_flow`
- `libero_pi05_subtask_fast`
- `libero_pi05_action_expert`
- `libero_pi05_subtask_hybrid`

These correspond to:

- flow-only action training
- subtask + FAST token training
- action-expert-only finetuning
- joint hybrid training

### 6. Evaluation and server/client stack added

The feature also includes runtime support for evaluating and serving the
subtask-capable Pi0.5 model:

- `examples/libero/main_subtask.py`
- `examples/libero/main_subtask_async.py`
- `scripts/async_pi05/`
- `scripts/sync_pi05_websocket_server.py`

This is why the feature footprint is large: it includes both training support and synchronous/asynchronous inference paths for LIBERO evaluation.

### 7. Additional imported files

Some files in this merge are related but not strictly required for the core
LIBERO subtask training path, for example:

- `src/openpi/policies/arx_policy.py`
- `src/openpi/policies/flexiv_new_policy.py`
- `src/openpi/policies/flexiv_subtask_policy.py`

Treat those as auxiliary imported work rather than the minimal core required for the subtask feature described in this README.

---

## Training Strategies

### Strategy A: Joint Training (All Three Losses)

Train all three loss components simultaneously. This uses
`libero_pi05_subtask_hybrid` in `src/openpi/training/config.py`:

```python
# Mode: Subtask + FAST + Flow (Hybrid — all three losses)
TrainConfig(
    name="libero_pi05_subtask_hybrid",
    exp_name="libero_subtask_hybrid",
    model=pi05_config.Pi05Config(
        action_horizon=20,
        max_token_len=192,
        discrete_state_input=False,
        subtask_loss_weight=0.15,
        fast_token_loss_weight=0.15,
        flow_matching_loss_weight=1.0,
        fast_tokenizer_path="physical-intelligence/fast",
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/home/kewang/.cache/openpi/openpi-assets/checkpoints/pi05_base/params"
    ),
    data=LeRobotLiberoSubtaskDataConfig(
        repo_id="KeWangRobotics/libero_10_subtasks",
        base_config=DataConfig(
            asset_id="libero_subtask",
            use_quantile_norm=True,  # Quantile normalization for gripper actions
        ),
    ),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=3000,
        peak_lr=2.5e-5,
        decay_steps=150_000,
        decay_lr=2.5e-6,
    ),
    num_train_steps=40_000,
    save_interval=5000,
    batch_size=64,
    fsdp_devices=1,
    ema_decay=0.999,
),
```

---

### Strategy B: Knowledge Insulation (Two Stages)

#### Stage 1 — Finetune the VLM (subtask + FAST token loss)

The VLM is finetuned while the action expert is frozen. Only subtask prediction
and discrete action token losses are used.

```python
# Mode: Subtask + FAST Token (discrete action tokens)
TrainConfig(
    name="libero_pi05_subtask_fast",
    exp_name="libero_subtask_fast",
    model=pi05_config.Pi05Config(
        action_horizon=25,
        max_token_len=256,
        discrete_state_input=False,
        subtask_loss_weight=10.0,
        fast_token_loss_weight=1.0,
        flow_matching_loss_weight=0.0,  # Disabled
        fast_tokenizer_path="physical-intelligence/fast",
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/home/kewang/.cache/openpi/openpi-assets/checkpoints/pi05_base/params"
    ),
    data=LeRobotLiberoSubtaskDataConfig(
        repo_id="KeWangRobotics/libero_10_subtasks",
        base_config=DataConfig(
            asset_id="libero_subtask",
            use_quantile_norm=True,
        ),
    ),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=3000,
        peak_lr=2.5e-5,
        decay_steps=150_000,
        decay_lr=2.5e-6,
    ),
    num_train_steps=20_000,
    save_interval=4000,
    batch_size=512,
    fsdp_devices=8,
    ema_decay=0.999,
    wandb_enabled=True,
),
```

#### Stage 2 — Finetune the Action Expert (flow matching loss only)

The VLM is frozen and only the action expert is trained using flow matching loss.
Gradients are blocked from the VLM via `freeze_filter`. The checkpoint is
initialized from Stage 1.

```python
# Mode: Action Expert only (flow matching)
TrainConfig(
    name="libero_pi05_action_expert",
    exp_name="libero_action_expert",
    model=pi05_config.Pi05Config(
        action_horizon=25,
        max_token_len=256,
        discrete_state_input=False,
        subtask_loss_weight=0.0,       # Disabled
        fast_token_loss_weight=0.0,    # Disabled
        flow_matching_loss_weight=1.0,
        fast_tokenizer_path="physical-intelligence/fast",
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/home/kewang/.cache/openpi/openpi-checkpoints/libero_pi05_subtask_fast/my_experiment/12000/params"
    ),
    data=LeRobotLiberoSubtaskDataConfig(
        repo_id="KeWangRobotics/libero_10_subtasks",
        base_config=DataConfig(
            asset_id="libero_subtask",
            use_quantile_norm=True,
        ),
    ),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=3000,
        peak_lr=2.5e-5,
        decay_steps=150_000,
        decay_lr=2.5e-6,
    ),
    num_train_steps=8_000,
    save_interval=2000,
    batch_size=512,
    fsdp_devices=8,
    ema_decay=0.999,
    wandb_enabled=True,
    freeze_filter=nnx.All(
        nnx.Param,
        nnx_utils.PathRegex(".*llm.*"),             # Freeze all LLM layers
        nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")), # Exclude action expert branch
    ),
),
```

---

## Setup

### 1. Download the FAST Tokenizer

```bash
uv run python scripts/download_fast_tokenizer.py
```

This stores the tokenizer in `weights/fast`, which matches the subtask training configs in `src/openpi/training/config.py`.
For cluster runs, copy that directory into the scratch-side `weights/` directory that gets bind-mounted into the repo.

### 2. Download the Pi0.5 Base Model

```bash
python - <<'PY'
from openpi.training import config as _config
from openpi.shared import download

config = _config.get_config("pi05_base")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_base")
PY
```

---

## Running Training

### Option A: Joint Training

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py libero_pi05_subtask_hybrid \
  --exp-name=my_experiment_all \
  --overwrite
```

### Option B: Knowledge Insulation

**Phase 1** — Finetune the VLM:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py libero_pi05_subtask_fast \
  --exp-name=my_experiment_all \
  --overwrite
```

**Phase 2** — Finetune the action expert (resume from Phase 1 checkpoint):

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py libero_pi05_action_expert \
  --exp-name=my_experiment_all \
  --overwrite
```

---

## Evaluation

First, see the [LIBERO README](README.md) to set up the environment.

### Synchronous Server + Client

**Start the Pi0.5 server:**

```bash
export OPENPI_DATA_HOME=$HOME/.cache/openpi
python scripts/async_pi05/sync_pi05_websocket_server.py \
  --config libero_pi05_action_expert \
  --checkpoint PATH_TO_CHECKPOINT \
  --gpu-id 0 \
  --host 0.0.0.0 \
  --port 8765
```

**Start the LIBERO client:**

```bash
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
python examples/libero/main_subtask.py --host 127.0.0.1 --port 8765
```

---

### Asynchronous Server + Client

Use the async stack for true non-blocking inference.

**Start the async Pi0.5 server:**

```bash
export OPENPI_DATA_HOME=$HOME/.cache/openpi
python scripts/async_pi05/async_pi05_websocket_server.py \
  --config libero_pi05_action_expert \
  --checkpoint PATH_TO_CHECKPOINT \
  --gpu-id 0 \
  --host 0.0.0.0 \
  --port 8765
```

**Test with a single query:**

```bash
python scripts/async_pi05/async_pi05_client.py \
  --host 127.0.0.1 \
  --port 8765 \
  --high-level-prompt "Pick up the flashcard on the table"
```

**Run the async LIBERO evaluation:**

```bash
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
python examples/libero/main_subtask_async.py --host 127.0.0.1 --port 8765
```
