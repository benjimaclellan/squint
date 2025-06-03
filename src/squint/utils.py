# Copyright 2024-2025 Benjamin MacLellan

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree


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


def partition_op(pytree: PyTree, name: str):  # TODO: allow multiple names
    """
    Partition a PyTree into parameters and static parts based on the operation name key.
    Args:
        pytree (PyTree): The input PyTree containing operations.
        name (str): The operation name key to filter by.
    """

    def select(pytree: PyTree, name: str):
        """Sets all leaves to `True` for a given op key from the given Pytree)"""
        get_leaf = lambda t: t.ops[name]
        null = jax.tree_util.tree_map(lambda _: True, pytree.ops[name])
        return eqx.tree_at(get_leaf, pytree, null)

    def mask(val: str, mask1, mask2):
        """Logical AND mask over Pytree"""
        if isinstance(mask1, bool) and isinstance(mask2, bool):
            if mask1 and mask2:
                return True
        else:
            return False

    _params = eqx.filter(pytree, eqx.is_inexact_array, inverse=True, replace=True)
    _op = select(pytree, name)

    filter = jax.tree_util.tree_map(mask, pytree, _params, _op)

    params, static = eqx.partition(pytree, filter_spec=filter)

    return params, static


def extract_paths(obj: PyTree, path="", op_type=None):
    """
    Recursively extract paths to non-None leaves in a PyTree, including their operation type.
    """
    if isinstance(obj, eqx.Module):
        op_type = type(
            obj
        ).__name__  # Capture the operation type (e.g., "BeamSplitter")
        for field_name in obj.__dataclass_fields__:  # Traverse dataclass fields
            field_value = getattr(obj, field_name)
            yield from extract_paths(
                field_value, f"{path}.{field_name}" if path else field_name, op_type
            )
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from extract_paths(v, f"{path}.{k}" if path else str(k), op_type)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from extract_paths(v, f"{path}[{i}]" if path else f"[{i}]", op_type)
    else:
        if obj is not None:
            yield path, op_type, obj  # Always return op_type, even if it's unchanged
