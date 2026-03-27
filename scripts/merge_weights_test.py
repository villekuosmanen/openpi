"""Tests for the merge_weights script."""

import os

import pytest
import numpy as np
from flax import traverse_util

os.environ["JAX_PLATFORMS"] = "cpu"

from . import merge_weights


def test_average_pytrees_equal_weighting():
    """Test weight averaging with equal weighting (0.5)."""
    tree1 = {
        "layer1": {"weight": np.array([1.0, 2.0, 3.0])},
        "layer2": {"bias": np.array([0.5, 0.5])},
    }
    tree2 = {
        "layer1": {"weight": np.array([3.0, 4.0, 5.0])},
        "layer2": {"bias": np.array([1.5, 1.5])},
    }
    
    merged = merge_weights.average_pytrees(tree1, tree2, alpha=0.5)
    
    # Check layer1 weight: 0.5 * [1, 2, 3] + 0.5 * [3, 4, 5] = [2, 3, 4]
    np.testing.assert_allclose(
        merged["layer1"]["weight"],
        np.array([2.0, 3.0, 4.0]),
    )
    
    # Check layer2 bias: 0.5 * [0.5, 0.5] + 0.5 * [1.5, 1.5] = [1.0, 1.0]
    np.testing.assert_allclose(
        merged["layer2"]["bias"],
        np.array([1.0, 1.0]),
    )


def test_average_pytrees_weighted():
    """Test weight averaging with custom weighting."""
    tree1 = {
        "layer": {"weight": np.array([0.0, 0.0])},
    }
    tree2 = {
        "layer": {"weight": np.array([10.0, 10.0])},
    }
    
    # Test with alpha=0.7 (70% tree1, 30% tree2)
    merged = merge_weights.average_pytrees(tree1, tree2, alpha=0.7)
    
    # Expected: 0.7 * [0, 0] + 0.3 * [10, 10] = [3, 3]
    np.testing.assert_allclose(
        merged["layer"]["weight"],
        np.array([3.0, 3.0]),
    )
    
    # Test with alpha=0.0 (pure tree2)
    merged = merge_weights.average_pytrees(tree1, tree2, alpha=0.0)
    np.testing.assert_allclose(
        merged["layer"]["weight"],
        tree2["layer"]["weight"],
    )
    
    # Test with alpha=1.0 (pure tree1)
    merged = merge_weights.average_pytrees(tree1, tree2, alpha=1.0)
    np.testing.assert_allclose(
        merged["layer"]["weight"],
        tree1["layer"]["weight"],
    )


def test_average_pytrees_nested():
    """Test weight averaging with deeply nested structures."""
    tree1 = {
        "module": {
            "submodule1": {"weight": np.array([1.0])},
            "submodule2": {"bias": np.array([2.0])},
        }
    }
    tree2 = {
        "module": {
            "submodule1": {"weight": np.array([5.0])},
            "submodule2": {"bias": np.array([6.0])},
        }
    }
    
    merged = merge_weights.average_pytrees(tree1, tree2, alpha=0.5)
    
    np.testing.assert_allclose(
        merged["module"]["submodule1"]["weight"],
        np.array([3.0]),
    )
    np.testing.assert_allclose(
        merged["module"]["submodule2"]["bias"],
        np.array([4.0]),
    )


def test_average_pytrees_structure_mismatch():
    """Test that mismatched structures raise an error."""
    tree1 = {
        "layer1": {"weight": np.array([1.0])},
    }
    tree2 = {
        "layer2": {"weight": np.array([2.0])},  # Different key
    }
    
    with pytest.raises(ValueError, match="Trees have different structures"):
        merge_weights.average_pytrees(tree1, tree2, alpha=0.5)


def test_average_pytrees_shape_mismatch():
    """Test that mismatched shapes raise an error."""
    tree1 = {
        "layer": {"weight": np.array([1.0, 2.0])},
    }
    tree2 = {
        "layer": {"weight": np.array([3.0, 4.0, 5.0])},  # Different shape
    }
    
    with pytest.raises(ValueError, match="Shape mismatch"):
        merge_weights.average_pytrees(tree1, tree2, alpha=0.5)


def test_average_pytrees_dtype_handling():
    """Test that dtype mismatches are handled correctly."""
    tree1 = {
        "layer": {"weight": np.array([1.0, 2.0], dtype=np.float32)},
    }
    tree2 = {
        "layer": {"weight": np.array([3.0, 4.0], dtype=np.float64)},
    }
    
    # Should convert tree2 to match tree1's dtype
    merged = merge_weights.average_pytrees(tree1, tree2, alpha=0.5)
    
    assert merged["layer"]["weight"].dtype == np.float32
    np.testing.assert_allclose(
        merged["layer"]["weight"],
        np.array([2.0, 3.0], dtype=np.float32),
    )


def test_average_pytrees_preserves_structure():
    """Test that the merged tree has the same structure as inputs."""
    tree1 = {
        "a": {"b": {"c": np.array([1.0])}},
        "d": {"e": np.array([2.0])},
    }
    tree2 = {
        "a": {"b": {"c": np.array([3.0])}},
        "d": {"e": np.array([4.0])},
    }
    
    merged = merge_weights.average_pytrees(tree1, tree2, alpha=0.5)
    
    # Check structure is preserved
    assert set(merged.keys()) == {"a", "d"}
    assert set(merged["a"].keys()) == {"b"}
    assert set(merged["a"]["b"].keys()) == {"c"}
    assert set(merged["d"].keys()) == {"e"}
    
    # Check values
    np.testing.assert_allclose(merged["a"]["b"]["c"], np.array([2.0]))
    np.testing.assert_allclose(merged["d"]["e"], np.array([3.0]))


def test_average_pytrees_with_flatten_unflatten():
    """Test that flattening and unflattening preserves behavior."""
    tree1 = {
        "layer1": {"weight": np.array([1.0, 2.0])},
        "layer2": {"bias": np.array([3.0])},
    }
    tree2 = {
        "layer1": {"weight": np.array([5.0, 6.0])},
        "layer2": {"bias": np.array([7.0])},
    }
    
    # Merge directly
    merged_direct = merge_weights.average_pytrees(tree1, tree2, alpha=0.3)
    
    # Merge via flatten/unflatten (simulating what the function does internally)
    flat1 = traverse_util.flatten_dict(tree1, sep="/")
    flat2 = traverse_util.flatten_dict(tree2, sep="/")
    flat_merged = {k: 0.3 * flat1[k] + 0.7 * flat2[k] for k in flat1}
    merged_flatten = traverse_util.unflatten_dict(flat_merged, sep="/")
    
    # Should give same results
    np.testing.assert_allclose(
        merged_direct["layer1"]["weight"],
        merged_flatten["layer1"]["weight"],
    )
    np.testing.assert_allclose(
        merged_direct["layer2"]["bias"],
        merged_flatten["layer2"]["bias"],
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
