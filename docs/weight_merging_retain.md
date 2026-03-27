# Weight Merging for Robust Policy Finetuning (RETAIN)

## Overview

This document describes the weight merging approach for robust policy finetuning, which improves both:
1. **Generalization** to unseen variations of the target task
2. **Retention** of generalist capabilities on non-target tasks

## Background

When fine-tuning generalist robot policies on limited demonstrations of a new task, models often:
- Overfit to the specific demonstrations
- Lose their prior generalist abilities
- Fail to generalize within the new task itself

The **RETAIN** approach addresses this by interpolating (averaging) the weights of a fine-tuned model with the pretrained base model. This simple yet effective strategy produces a single model that inherits the generalist abilities of the base model while learning to solve the new task robustly.

## How It Works

The approach is straightforward: given a pretrained base model with weights `W_base` and a fine-tuned model with weights `W_finetuned`, we create merged weights:

```
W_merged = α × W_base + (1 - α) × W_finetuned
```

where `α` is the merge ratio (typically 0.5 for equal weighting).

### Why Does This Work?

1. **Weight Space Interpolation**: Neural network weights can be meaningfully interpolated in weight space when models share the same architecture and initialization
2. **Regularization Effect**: The base model acts as a regularizer, preventing the fine-tuned model from drifting too far from the general-purpose solution
3. **Knowledge Preservation**: The base model's knowledge is explicitly preserved in the merged weights

## Usage

### Basic Usage

```bash
python scripts/merge_weights.py \
    --finetuned_checkpoint_path ./checkpoints/pi05_bin_pack/exp1/20000/params \
    --config_name pi05_bin_pack_coffee_capsules \
    --output_directory ./checkpoints/merged_weights \
    --merge_ratio 0.5
```

### Parameters

- `--finetuned_checkpoint_path`: Path to the fine-tuned checkpoint params directory
  - Example: `./checkpoints/pi05_bin_pack/exp1/20000/params`
  - Can also be a GCS path: `gs://my-bucket/checkpoints/params`

- `--config_name`: Name of the training config used for fine-tuning
  - This is used to automatically identify the base checkpoint
  - Must match one of the configs in `src/openpi/training/config.py`
  - Examples: `pi05_bin_pack_coffee_capsules`, `pi0_libero`, `pi05_droid_finetune`

- `--output_directory`: Directory where the merged checkpoint will be saved
  - The script creates this directory if it doesn't exist
  - Can be used directly for inference or further fine-tuning

- `--merge_ratio`: Weight for the base model (default: 0.5)
  - `0.5`: Equal weighting (recommended starting point)
  - `0.0`: Pure fine-tuned model (no merging)
  - `1.0`: Pure base model (no fine-tuning)
  - `0.3`: 30% base, 70% fine-tuned (more task-specific)
  - `0.7`: 70% base, 30% fine-tuned (more generalist)

- `--base_checkpoint_path`: (Optional) Override the base checkpoint path
  - By default, uses the checkpoint specified in the training config
  - Use this to merge with a different base checkpoint

- `--verbose`: Enable verbose logging for debugging

## Choosing the Merge Ratio

The optimal merge ratio depends on your use case:

| Merge Ratio (α) | Use Case | Trade-off |
|-----------------|----------|-----------|
| 0.5 | **Recommended default** | Balanced task performance and generalization |
| 0.3 - 0.4 | Strong task-specific performance needed | Better on target task, less generalization |
| 0.6 - 0.7 | Strong generalist capabilities needed | Better generalization, slightly lower task performance |
| 0.0 | Pure fine-tuning baseline (for comparison) | High task performance on seen demos, poor generalization |
| 1.0 | Pure base model baseline (for comparison) | Full generalist capabilities, no task improvement |

### Recommendation

1. Start with `merge_ratio=0.5` (equal weighting)
2. Evaluate on both:
   - **Target task variations**: Test generalization to unseen scenarios
   - **Non-target tasks**: Test retention of generalist capabilities
3. Adjust based on results:
   - If target task performance is insufficient, decrease to 0.3-0.4
   - If generalization or retention is insufficient, increase to 0.6-0.7

## Examples

### Example 1: Bin Pack Task

```bash
# Fine-tune on bin pack task
python scripts/train.py pi05_bin_pack_coffee_capsules \
    --exp_name my_finetune_experiment

# Merge weights after training (e.g., at step 20000)
python scripts/merge_weights.py \
    --finetuned_checkpoint_path ./checkpoints/pi05_bin_pack_coffee_capsules/my_finetune_experiment/20000/params \
    --config_name pi05_bin_pack_coffee_capsules \
    --output_directory ./checkpoints/bin_pack_merged \
    --merge_ratio 0.5

# Use merged checkpoint for inference
# (Update your inference config to point to ./checkpoints/bin_pack_merged)
```

### Example 2: DROID Task

```bash
# Fine-tune on custom DROID dataset
python scripts/train.py pi05_droid_finetune \
    --exp_name droid_experiment

# Merge with different ratios to find best balance
for ratio in 0.3 0.5 0.7; do
    python scripts/merge_weights.py \
        --finetuned_checkpoint_path ./checkpoints/pi05_droid_finetune/droid_experiment/20000/params \
        --config_name pi05_droid_finetune \
        --output_directory ./checkpoints/droid_merged_${ratio} \
        --merge_ratio ${ratio}
done

# Evaluate each merged model and pick the best
```

### Example 3: LIBERO Task

```bash
# Fine-tune on LIBERO
python scripts/train.py pi0_libero \
    --exp_name libero_experiment

# Merge weights
python scripts/merge_weights.py \
    --finetuned_checkpoint_path ./checkpoints/pi0_libero/libero_experiment/30000/params \
    --config_name pi0_libero \
    --output_directory ./checkpoints/libero_merged \
    --merge_ratio 0.5 \
    --verbose
```

## Implementation Details

### Weight Loading

The script uses the `restore_params` function from `openpi.models.model`, which:
- Handles both local and GCS paths
- Automatically downloads GCS checkpoints
- Returns weights as numpy arrays for efficient averaging

### Weight Averaging

The averaging is performed in the flattened weight space:
1. Both weight trees are flattened to dictionaries with path keys
2. Each weight tensor is averaged independently: `w_merged = α × w_base + (1 - α) × w_finetuned`
3. The flattened dictionary is reconstructed back into the nested pytree structure

### Error Checking

The script validates:
- Both checkpoints have the same structure (same keys)
- Corresponding weights have matching shapes
- Corresponding weights have compatible dtypes (auto-converts if needed)

### Saving

The merged checkpoint is saved using orbax's `PyTreeCheckpointer`, maintaining compatibility with:
- OpenPI inference scripts
- Further fine-tuning
- All existing checkpoint loading utilities

## Troubleshooting

### Error: "Trees have different structures"

**Cause**: The base and fine-tuned checkpoints have different parameter structures.

**Solutions**:
1. Verify both models use the same architecture (e.g., both Pi0, or both Pi0.5)
2. Check if you're using LoRA fine-tuning (which adds new parameters)
3. Ensure the fine-tuned model was initialized from the same base checkpoint

### Error: "Shape mismatch"

**Cause**: Corresponding parameters have different shapes.

**Solutions**:
1. Verify both models have the same `action_dim` and `action_horizon`
2. Check if the fine-tuned model was configured correctly
3. Ensure you're merging compatible model variants (e.g., both pi0_base, not pi0_base + pi0_fast)

### Warning: "Dtype mismatch"

**Cause**: Parameters have different data types (e.g., float32 vs bfloat16).

**Solution**: This is usually safe - the script automatically converts to match the base model dtype. If you see unexpected behavior, check your training config's `pytorch_training_precision` setting.

### Merged model performs poorly on both tasks

**Cause**: Merge ratio may need adjustment.

**Solutions**:
1. Try different merge ratios (0.3, 0.5, 0.7)
2. Verify the fine-tuned model itself performs well before merging
3. Check if the base checkpoint path is correct

## References

This implementation is based on the weight merging approach for robust policy finetuning. The core idea is that weight space interpolation between a pretrained model and a fine-tuned model can improve both:
- Out-of-distribution generalization on the target task
- Retention of pre-training capabilities on non-target tasks

## See Also

- `scripts/merge_weights.py` - Main implementation
- `scripts/merge_weights_example.sh` - Usage examples
- `src/openpi/training/config.py` - Training configurations
- `src/openpi/models/model.py` - Model loading utilities
