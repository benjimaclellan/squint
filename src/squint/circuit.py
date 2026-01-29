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

from beartype import beartype

from squint.ops.base import (
    Block,
)


class Circuit(Block):
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

    @beartype
    @classmethod
    def from_block(
        cls,
        block: Block,
    ):
        """Promote a Block to a Circuit"""
        self = cls()
        self.ops = block.ops
        return self
