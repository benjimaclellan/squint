# # Copyright 2024-2026 Benjamin MacLellan

# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at

# #     http://www.apache.org/licenses/LICENSE-2.0

# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# # %%
# import equinox as eqx
# from beartype import beartype
# import functools
# import itertools
# from collections import OrderedDict
# from typing import Optional, Union

# import equinox as eqx
# import jax.numpy as jnp
# import scipy as sp
# from beartype import beartype
# from beartype.door import is_bearable
# from beartype.typing import Callable, Sequence
# from ordered_set import OrderedSet

# from squint.ops.gellmann import gellmann

# _wire_id = itertools.count(1)

# # from squint.ops.base import (
# #     Block,
# # )


# class Circuit(eqx.Module):
#     """
#     A block operation that groups a sequence of quantum operations.

#     Blocks allow organizing multiple operations into a single logical unit.
#     They can be nested within circuits or other blocks, and support the same
#     `add` and `unwrap` interface as Circuit. Unlike Circuit, Block does not
#     specify a backend and is purely for organizational purposes.

#     Attributes:
#         ops (OrderedDict): Ordered dictionary mapping keys to operations or nested blocks.

#     Example:
#         ```python
#         from squint.ops.base import Block, Wire
#         from squint.ops.dv import RXGate, RYGate

#         wire = Wire(dim=2, idx=0)
#         block = Block()
#         block.add(RXGate(wires=(wire,), phi=0.1), "rx")
#         block.add(RYGate(wires=(wire,), phi=0.2), "ry")

#         # Use in a circuit
#         circuit.add(block, "rotation_block")
#         ```
#     """

#     ops: OrderedDict[Union[str, int], Union[AbstractOp, "Circuit"]]
#     # ops: dict[Union[str, int], Union[AbstractOp, "Block"]]

#     @beartype
#     def __init__(
#         self,
#         ops: dict | OrderedDict = {}
#         # ops: OrderedDict = OrderedDict()
#     ):
#         """
#         Initialize an empty Block.

#         Creates a new Block with no operations. Operations can be added
#         using the `add` method.
#         """
#         self.ops = OrderedDict(ops)

#     @property
#     def wires(self) -> Sequence[Wire]:
#         """
#         Get all wires used by operations in this block.

#         Returns:
#             set[Wire]: Set of all Wire objects that operations in this block act on.
#         """
#         # BUG: this line caused a bug with undefined wire order
#         # return set(sum((op.wires for op in self.unwrap()), ()))
#         return OrderedSet(
#             sorted(
#                 dict.fromkeys(
#                     itertools.chain.from_iterable(op.wires for op in self.unwrap())
#                 ),
#                 key=wire_sort_key,
#             )
#         )
        
#     @beartype
#     def add(self, op: Union[AbstractOp, "Circuit"], key: str = None) -> None:
#         """
#         Add an operator to the block.

#         Operators are added sequentially. When this block is used in a circuit,
#         the operations will be applied in the order they were added.

#         Args:
#             op (AbstractOp | Block): The operator or nested block to add.
#             key (str, optional): A string key for indexing into the block's ops
#                 dictionary. If None, an integer counter is used as the key.
#         """

#         if key is None:
#             key = len(self.ops)
#         self.ops[key] = op

#     # def unwrap(self) -> tuple[AbstractOp]:
#     #     """
#     #     Unwrap all operators in the block into a flat tuple.

#     #     Recursively calls `unwrap()` on all contained operations and nested
#     #     blocks to produce a flat sequence of atomic operations.

#     #     Returns:
#     #         tuple[AbstractOp]: Flattened tuple of all operations in order.
#     #     """
#     #     return tuple(
#     #         op for op_wrapped in self.ops.values() for op in op_wrapped.unwrap()
#     #     )
#     #     # return Block(
#     #     #     ops=
#     #     #     {k: op for k, op_wrapped in self.ops.values() for op in op_wrapped.unwrap()
#     #     # )



# # class Circuit(Block):
# #     r"""
# #     The `Circuit` object is a symbolic representation of a quantum circuit for qubits, qudits, or for an infinite-dimensional Fock space.
# #     The circuit is composed of a sequence of quantum operators on `wires` which define the evolution of the quantum

# #     Attributes:
# #         ops (dict[Union[str, int], AbstractOp]): A dictionary of ops (dictionary value) with an assigned label (dictionary key).

# #     Example:
# #         ```python
# #         circuit = Circuit(backend='pure')
# #         circuit.add(DiscreteVariableState(wires=(0,)))
# #         circuit.add(HGate(wires=(0,)))
# #         ```
# #     """

# #     @beartype
# #     @classmethod
# #     def from_block(
# #         cls,
# #         block: Block,
# #     ):
# #         """Promote a Block to a Circuit"""
# #         self = cls()
# #         self.ops = block.ops
# #         return self
