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
from beartype.typing import Union, Sequence
from beartype.door import is_bearable

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


def partition_op(pytree: PyTree, name: Union[str, Sequence[str]]):  # TODO: allow multiple names
    """
    Partition a PyTree into parameters and static parts based on the operation name key.
    Args:
        pytree (PyTree): The input PyTree containing operations.
        name (str): The operation name key to filter by.
    """
    # if isinstance(names, str):
        # names = [names]
        
        
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
    # _ops = [select(pytree, name) for name in names]
    _op = select(pytree, name)

    # filter = jax.tree_util.tree_map(mask, pytree, _params, *_ops)
    filter = jax.tree_util.tree_map(mask, pytree, _params, _op)

    params, static = eqx.partition(pytree, filter_spec=filter)

    return params, static



# import jax
# import equinox as eqx
# from typing import Union, Sequence
# from jaxtyping import PyTree
# from collections.abc import Callable

# def partition_op(pytree: PyTree, names: Union[str, Sequence[str]]):
#     """
#     Partition a PyTree into parameters and static parts based on one or more operation name keys.
#     Args:
#         pytree (PyTree): The input PyTree containing operations.
#         names (Union[str, Sequence[str]]): One or more operation name keys to filter by.
#     """
#     if isinstance(names, str):
#         names = [names]

#     # Step 1: Get a mask of all inexact array leaves (trainable parameters)
#     is_param = eqx.filter(pytree, eqx.is_inexact_array, inverse=True, replace=True)

#     # Step 2: Build a mask that is True for leaves under any of the selected op names
#     def is_op_leaf(path):
#         found_ops = False
#         found_match = False
#         for p in path:
#             if isinstance(p, eqx.GetAttrKey) and p.name == "ops":
#                 found_ops = True
#             if found_ops and isinstance(p, eqx.DictKey) and p.key in names:
#                 found_match = True
#         return found_match

#     def mask_fn(path, leaf):
#         return is_op_leaf(path)

#     is_op_mask = jax.tree_util.tree_map_with_path(mask_fn, pytree, is_leaf=lambda _: True)

#     # Step 3: Combine masks — True if both are True
#     filter_spec = jax.tree_util.tree_map(lambda a, b: a and b, is_param, is_op_mask)

#     # Step 4: Partition based on combined mask
#     return eqx.partition(pytree, filter_spec=filter_spec)



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
