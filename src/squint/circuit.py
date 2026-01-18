# Copyright 2024-2026 Benjamin MacLellan

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
import functools
import itertools
import warnings
from collections import OrderedDict
from typing import Literal, Optional, Sequence, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import paramax
from beartype import beartype
from jaxtyping import PyTree
from opt_einsum.parser import get_symbol

from squint.ops.base import (
    AbstractErasureChannel,
    AbstractKrausChannel,
    AbstractMixedState,
    AbstractPureState,
    Block,
)
# from squint.simulator import (
#     Simulator,
#     SimulatorClassicalProbabilities,
#     SimulatorQuantumAmplitudes,
#     classical_fisher_information_matrix,
#     quantum_fisher_information_matrix,
# )

# from squint.simulators.tn import _compile, _subscripts_pure, _subscripts_mixed


class Circuit(Block):
    # class Circuit(eqx.Module):
    r"""
    The `Circuit` object is a symbolic representation of a quantum circuit for qubits, qudits, or for an infinite-dimensional Fock space.
    The circuit is composed of a sequence of quantum operators on `wires` which define the evolution of the quantum

    Attributes:
        ops (dict[Union[str, int], AbstractOp]): A dictionary of ops (dictionary value) with an assigned label (dictionary key).

    Example:
        ```python
        circuit = Circuit(backend='pure')
        circuit.add(DiscreteVariableState(wires=(0,)))
        circuit.add(HGate(wires=(0,)))
        ```
    """

    # ops: OrderedDict[Union[str, int], Union[AbstractOp, Block]]
    # _backend: Literal["pure", "mixed"]

    @beartype
    # def __init__(self, backend: Optional[Literal["pure", "mixed"]] = None):
    def __init__(self):
        """
        Initializes a quantum circuit with the specified backend type.

        Args:
            backend (Literal["pure", "mixed"]): The type of backend to use for the circuit.
            Defaults to "pure". "pure" represents a reversible quantum operation,
            while "mixed" allows for non-reversible operations.
        """
        self.ops = OrderedDict()
        # self._backend = backend

    @beartype
    @classmethod
    def from_block(
        cls, 
        block: Block, 
        # backend: Optional[Literal["pure", "mixed"]] = None
    ):
        """Promote a Block to a Circuit"""
        self = cls()
        # self = cls(backend=backend)
        self.ops = block.ops
        return self

    # @property
    # def wires(self) -> set[int]:
    #     """
    #     Initializes a quantum circuit with the specified backend type.

    #     Args:
    #         backend (Literal["pure", "mixed"]): The type of backend to use for the circuit.
    #         Defaults to "pure". "pure" represents a reversible quantum operation,
    #         while "mixed" allows for non-reversible operations.
    #     """
    #     return set(sum((op.wires for op in self.unwrap()), ()))

    # @beartype
    # def add(self, op: Union[AbstractOp, Block], key: str = None) -> None:
    #     """
    #     Add an operator to the circuit.

    #     Operators are added sequential along the wires. The first operator on each wire must be a state
    #     (a subtype of AbstractPureState or AbstractMixedState).

    #     Args:
    #         op (AbstractOp): The operator instance to add to the circuit.
    #         key (Optional[str]): A string key for indexing into the circuit PyTree instance. Defaults to `None` and an integer counter is used.
    #     """

    #     if key is None:
    #         key = len(self.ops)
    #     self.ops[key] = op

    # def unwrap(self) -> tuple[AbstractOp]:
    #     """
    #     Unwrap all operators in the circuit by recursively calling the `op.unwrap()` method.
    #     """
    #     return tuple(
    #         op for op_wrapped in self.ops.values() for op in op_wrapped.unwrap()
    #     )


    # @property
    # def backend(self) -> str:
    #     if self._backend == None:
    #         if any(
    #             [
    #                 isinstance(
    #                     op,
    #                     (
    #                         AbstractMixedState,
    #                         AbstractKrausChannel,
    #                         AbstractErasureChannel,
    #                     ),
    #                 )
    #                 for op in self.unwrap()
    #             ]
    #         ):
    #             return "mixed"
    #         else:
    #             return "pure"
    #     return self._backend

    # @property
    # def subscripts(self) -> str:
    #     """
    #     Returns the einsum subscript expression as a string.
    #     """
    #     if self.backend == "mixed":
    #         return _subscripts_mixed(self)
    #     elif self.backend == "pure":
    #         return _subscripts_pure(self)

    # @beartype
    # def path(
    #     self,
    #     optimize: str = "greedy",
    # ):
    #     """
    #     Computes the einsum contraction path using the `opt_einsum` algorithm.

    #     Args:
    #         optimize (str): The argument to pass to `opt_einsum` for computing the optimal contraction path. Defaults to `greedy`.
    #     """

    #     path, info = jnp.einsum_path(
    #         self.subscripts,
    #         *self.evaluate(),
    #         optimize=optimize,
    #     )
    #     return path, info

    # @beartype
    # def evaluate(
    #     self,
    # ):
    #     """
    #     Evaluates the corresponding numerical tensor for each operator in the circuit, based on the provided dimension.
    #     """
    #     if self.backend == "pure":
    #         return [op() for op in self.unwrap()]

    #     elif self.backend == "mixed":
    #         _tensors = []
    #         for op in self.unwrap():
    #             _tensor = op()
    #             if isinstance(op, AbstractMixedState):
    #                 _tensors.append(_tensor)
    #             else:
    #                 # unconjugated/right + conj/left direction of tensor network, sequential in the list
    #                 _tensors.append(_tensor)
    #                 _tensors.append(jnp.conjugate(_tensor))
    #         return _tensors

    # def verify(self):
    #     """
    #     Performs a verification check on the circuit object to ensure it is valid prior to being compiled.
    #     """
    #     grid = {}
    #     for op in self.unwrap():
    #         for wire in op.wires:
    #             if wire not in grid.keys():
    #                 grid[wire] = []
    #             grid[wire].append(op)

    #     # check that the first op on each wire is an AbstractState, and no others are AbstractState ops
    #     for wire, ops in grid.items():
    #         if not isinstance(ops[0], (AbstractPureState, AbstractMixedState)):
    #             raise RuntimeError(
    #                 f"The first op on wire {wire} is of type {type(ops[0])}"
    #                 "The first op on each wire must be a subtype of `AbstractPureState` or `AbstractMixedState"
    #             )
    #         if any(
    #             [
    #                 isinstance(op, (AbstractPureState, AbstractMixedState))
    #                 for op in ops[1:]
    #             ]
    #         ):
    #             raise RuntimeError(
    #                 f"Wire {wire} contains multiple `AbstractState` ops."
    #                 "Only the first op on each wire can be a subtype of `AbstractPureState` or `AbstractMixedState"
    #             )

    #     # check that we are using the correct backend
    #     if any(
    #         [
    #             isinstance(
    #                 op,
    #                 (AbstractKrausChannel, AbstractErasureChannel, AbstractMixedState),
    #             )
    #             for op in self.unwrap()
    #         ]
    #     ):
    #         _backend = "mixed"
    #         if self.backend != _backend:
    #             raise RuntimeError(
    #                 "Backend must be `mixed` as the circuit contains one or more `AbstractChannel` and/or `AbstractMixedState`"
    #             )
    #     else:
    #         _backend = "pure"
    #         if self.backend != _backend:
    #             warnings.warn(
    #                 f"Circuit backend is set to `{self.backend}`; however the circuit is `pure`."
    #                 "Consider switching the backend to `pure`.",
    #                 UserWarning,
    #                 stacklevel=2,
    #             )

    # @staticmethod
    # def compile(
    #     static: PyTree,
    #     *params,
    #     **kwargs,
    # ) -> Simulator:
    #     """
    #     Compiles the circuit into a tensor contraction function.
    #
    #     Args:
    #         static (PyTree): The static PyTree, following the `equinox` convention. These are parameters that are fixed.
    #         params (Sequence[PyTree]): The parameterized PyTree, following the `equinox` convention. These are parameters that will be used in gradient and Fisher information calculations.
    #
    #     Returns:
    #         sim (Simulator): A class which contains methods for computing the parameterized forward, grad, and Fisher information functions.
    #     """
    #     return _compile(static, *params, **kwargs)
