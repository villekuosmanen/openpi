#!/bin/bash
# Example usage of the weight merging script for robust policy finetuning

# Example 1: Merge weights with equal weighting (0.5)
# This gives equal weight to both base and fine-tuned models
python scripts/merge_weights.py \
    --finetuned_checkpoint_path ./checkpoints/pi05_bin_pack_coffee_capsules/my_experiment/20000/params \
    --config_name pi05_bin_pack_coffee_capsules \
    --output_directory ./checkpoints/merged_equal_weight \
    --merge_ratio 0.5

# Example 2: Merge with more weight on fine-tuned model (0.3 base, 0.7 fine-tuned)
# Use this if you want to retain more of the fine-tuned behavior
python scripts/merge_weights.py \
    --finetuned_checkpoint_path ./checkpoints/pi05_bin_pack_coffee_capsules/my_experiment/20000/params \
    --config_name pi05_bin_pack_coffee_capsules \
    --output_directory ./checkpoints/merged_more_finetuned \
    --merge_ratio 0.3

# Example 3: Merge with more weight on base model (0.7 base, 0.3 fine-tuned)
# Use this if you want to retain more generalist capabilities
python scripts/merge_weights.py \
    --finetuned_checkpoint_path ./checkpoints/pi05_bin_pack_coffee_capsules/my_experiment/20000/params \
    --config_name pi05_bin_pack_coffee_capsules \
    --output_directory ./checkpoints/merged_more_base \
    --merge_ratio 0.7

# Example 4: Override base checkpoint path manually
# Use this if you want to merge with a different base checkpoint
python scripts/merge_weights.py \
    --finetuned_checkpoint_path ./checkpoints/pi05_bin_pack_coffee_capsules/my_experiment/20000/params \
    --config_name pi05_bin_pack_coffee_capsules \
    --base_checkpoint_path gs://openpi-assets/checkpoints/pi05_base/params \
    --output_directory ./checkpoints/merged_custom_base \
    --merge_ratio 0.5

# Example 5: Merge DROID fine-tuned model
python scripts/merge_weights.py \
    --finetuned_checkpoint_path ./checkpoints/pi05_droid_finetune/my_experiment/20000/params \
    --config_name pi05_droid_finetune \
    --output_directory ./checkpoints/pi05_droid_merged \
    --merge_ratio 0.5

# Example 6: Merge LIBERO fine-tuned model
python scripts/merge_weights.py \
    --finetuned_checkpoint_path ./checkpoints/pi0_libero/my_experiment/20000/params \
    --config_name pi0_libero \
    --output_directory ./checkpoints/pi0_libero_merged \
    --merge_ratio 0.5 \
    --verbose
