import jax.numpy as jnp

from squint.utils.partition import (
    extract_paths,
    partition_by_branches,
    partition_by_leaf_names,
    partition_by_leaves,
    partition_op,
)

__all__ = [
    "partition_op",
    "extract_paths",
    "print_nonzero_entries",
    "partition_by_leaves",
    "partition_by_branches",
    "partition_by_leaf_names",
]


def print_nonzero_entries(arr):
    """
    Print the indices and values of non-zero entries in a JAX array.
    Args:
        arr (jnp.ndarray): The JAX array to inspect.
    """
    nonzero_indices = jnp.array(jnp.nonzero(arr)).T
    nonzero_values = arr[tuple(nonzero_indices.T)]
    for idx, value in zip(nonzero_indices, nonzero_values, strict=True):
        print(f"Basis: {jnp.array(idx)}, Value: {value}")
