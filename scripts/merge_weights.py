#!/usr/bin/env python3
"""Merge fine-tuned model weights with base model weights using weight averaging.

This implements the RETAIN approach for robust policy finetuning:
By averaging the generalist policy weights before and after finetuning in weight space,
we obtain finetuned policies that:
(1) significantly improve generalization to unseen variations of the target task, and
(2) retain generalist capabilities on non-target tasks.

Usage:
    python scripts/merge_weights.py \
        --finetuned_checkpoint_path ./checkpoints/pi05_bin_pack/exp1/20000/params \
        --config_name pi05_bin_pack_coffee_capsules \
        --output_directory ./checkpoints/merged_weights \
        --merge_ratio 0.5

Args:
    finetuned_checkpoint_path: Path to the fine-tuned checkpoint params directory.
    config_name: Name of the training config (used to identify the base checkpoint).
    output_directory: Directory where the merged checkpoint will be saved.
    merge_ratio: Weight for the base model in the average (0.5 = equal weight).
                 merge_ratio=0.0 gives only fine-tuned, 1.0 gives only base model.
"""

import argparse
import logging
import pathlib
from typing import Any

from etils import epath
import jax
import jax.numpy as jnp
from flax import traverse_util
import numpy as np
import orbax.checkpoint as ocp

from openpi.models import model as _model
from openpi.shared import download as _download
from openpi.training import config as train_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def average_pytrees(
    tree1: dict[str, Any],
    tree2: dict[str, Any],
    alpha: float = 0.5,
) -> dict[str, Any]:
    """Average two pytrees with the same structure.
    
    Args:
        tree1: First pytree (typically base model weights).
        tree2: Second pytree (typically fine-tuned model weights).
        alpha: Weight for tree1. Result = alpha * tree1 + (1 - alpha) * tree2.
               alpha=0.5 means equal averaging.
    
    Returns:
        A new pytree with averaged weights.
    """
    flat_tree1 = traverse_util.flatten_dict(tree1, sep="/")
    flat_tree2 = traverse_util.flatten_dict(tree2, sep="/")
    
    # Check that both trees have the same structure
    keys1 = set(flat_tree1.keys())
    keys2 = set(flat_tree2.keys())
    
    if keys1 != keys2:
        missing_in_tree1 = keys2 - keys1
        missing_in_tree2 = keys1 - keys2
        error_msg = []
        if missing_in_tree1:
            error_msg.append(f"Keys in tree2 but not in tree1: {missing_in_tree1}")
        if missing_in_tree2:
            error_msg.append(f"Keys in tree1 but not in tree2: {missing_in_tree2}")
        raise ValueError("Trees have different structures:\n" + "\n".join(error_msg))
    
    # Average the weights
    merged = {}
    for key in flat_tree1:
        w1 = flat_tree1[key]
        w2 = flat_tree2[key]
        
        # Check shapes match
        if w1.shape != w2.shape:
            raise ValueError(
                f"Shape mismatch for key {key}: tree1 has {w1.shape}, tree2 has {w2.shape}"
            )
        
        # Check dtypes match
        if w1.dtype != w2.dtype:
            logger.warning(
                f"Dtype mismatch for key {key}: tree1 has {w1.dtype}, tree2 has {w2.dtype}. "
                f"Converting to {w1.dtype}."
            )
            w2 = w2.astype(w1.dtype)
        
        # Perform weighted average
        merged[key] = alpha * w1 + (1 - alpha) * w2
        
        logger.debug(f"Averaged {key}: shape={merged[key].shape}, dtype={merged[key].dtype}")
    
    return traverse_util.unflatten_dict(merged, sep="/")


def load_checkpoint_params(checkpoint_path: str | pathlib.Path) -> dict[str, Any]:
    """Load parameters from a checkpoint directory.
    
    Args:
        checkpoint_path: Path to the params directory.
    
    Returns:
        Parameter pytree as numpy arrays.
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Download if it's a GCS path
    local_path = _download.maybe_download(str(checkpoint_path))
    
    # Use restore_params which handles both GCS and local paths
    params = _model.restore_params(local_path, restore_type=np.ndarray)
    
    logger.info(f"Successfully loaded checkpoint with {len(traverse_util.flatten_dict(params))} parameters")
    return params


def save_checkpoint_params(params: dict[str, Any], output_dir: str | pathlib.Path):
    """Save parameters to a checkpoint directory.
    
    Args:
        params: Parameter pytree to save.
        output_dir: Directory where to save the checkpoint.
    """
    output_path = epath.Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving merged checkpoint to {output_path}")
    
    # Save using orbax CheckpointHandler
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(
            output_path,
            {"params": params},
            force=True,
        )
    
    logger.info(f"Successfully saved merged checkpoint to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge fine-tuned model weights with base model weights using weight averaging."
    )
    parser.add_argument(
        "--finetuned_checkpoint_path",
        type=str,
        required=True,
        help="Path to the fine-tuned checkpoint params directory (e.g., ./checkpoints/exp/20000/params)",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        help="Name of the training config (e.g., pi05_bin_pack_coffee_capsules)",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        required=True,
        help="Directory where the merged checkpoint will be saved",
    )
    parser.add_argument(
        "--merge_ratio",
        type=float,
        default=0.5,
        help="Weight for the base model in the average. 0.5 = equal weight, 0.0 = only fine-tuned, 1.0 = only base",
    )
    parser.add_argument(
        "--base_checkpoint_path",
        type=str,
        default=None,
        help="Optional: Override the base checkpoint path instead of using the one from config",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate merge ratio
    if not 0.0 <= args.merge_ratio <= 1.0:
        raise ValueError(f"merge_ratio must be between 0.0 and 1.0, got {args.merge_ratio}")
    
    logger.info("=" * 80)
    logger.info("Weight Merging for Robust Policy Finetuning (RETAIN)")
    logger.info("=" * 80)
    logger.info(f"Fine-tuned checkpoint: {args.finetuned_checkpoint_path}")
    logger.info(f"Config name: {args.config_name}")
    logger.info(f"Merge ratio (base model weight): {args.merge_ratio}")
    logger.info(f"Output directory: {args.output_directory}")
    
    # Load the training config to get the base checkpoint path
    logger.info(f"\nLoading training config '{args.config_name}'...")
    cfg = train_config.get_config(args.config_name)
    
    # Get base checkpoint path
    if args.base_checkpoint_path:
        base_checkpoint_path = args.base_checkpoint_path
        logger.info(f"Using provided base checkpoint path: {base_checkpoint_path}")
    else:
        if not hasattr(cfg.weight_loader, 'params_path'):
            raise ValueError(
                f"Config '{args.config_name}' does not have a CheckpointWeightLoader. "
                f"Please specify --base_checkpoint_path manually."
            )
        base_checkpoint_path = cfg.weight_loader.params_path
        logger.info(f"Using base checkpoint from config: {base_checkpoint_path}")
    
    # Load both checkpoints
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Loading base model weights")
    logger.info("=" * 80)
    base_params = load_checkpoint_params(base_checkpoint_path)
    
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Loading fine-tuned model weights")
    logger.info("=" * 80)
    finetuned_params = load_checkpoint_params(args.finetuned_checkpoint_path)
    
    # Merge the weights
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Averaging weights")
    logger.info("=" * 80)
    logger.info(
        f"Computing: merged = {args.merge_ratio} * base + {1 - args.merge_ratio} * finetuned"
    )
    
    merged_params = average_pytrees(
        base_params,
        finetuned_params,
        alpha=args.merge_ratio,
    )
    
    # Save the merged checkpoint
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Saving merged checkpoint")
    logger.info("=" * 80)
    save_checkpoint_params(merged_params, args.output_directory)
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ Weight merging completed successfully!")
    logger.info("=" * 80)
    logger.info(f"Merged checkpoint saved to: {args.output_directory}")
    logger.info(
        "\nYou can now use this checkpoint for inference or further fine-tuning."
    )


if __name__ == "__main__":
    main()
