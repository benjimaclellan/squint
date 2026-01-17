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
from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Literal, Sequence, Type, Union

from squint.ops import dv
from squint.ops.base import AbstractGate, Block, Wire


@beartype
def mzi_mesh(wires: Sequence[Wire], ansatz: Literal["reck", "clement"]) -> Block:
    # TODO: Reck and Clement's MZI mesh
    block = Block()
    return block


def _chunk_pairs(
    x: tuple[Wire, ...], periodic: bool = False
) -> tuple[
    tuple[tuple[Wire, Wire], ...], tuple[tuple[Wire, Wire], ...]
]:
    """
    Split a sequence of wires into pairs for a brickwork block.
    If periodic is True, the last wire is paired with the first wire.
    Args:
        x (tuple[int, ...]): A tuple of wire indices.
        periodic (bool): Whether to use periodic boundary conditions.
    Returns:
        tuple: Two tuples containing pairs of wires.
    """
    n = len(x)

    first = tuple((x[i], x[i + 1]) for i in range(0, n - 1, 2))
    second = tuple((x[i], x[i + 1]) for i in range(1, n - 1, 2))

    if periodic and n > 1 and n % 2 == 0:
        second += ((x[-1], x[0]),)

    return first, second


# %%
@beartype
def brickwork(
    wires: Sequence[Wire],
    depth: int,
    LocalGates: Union[Type[AbstractGate], Sequence[Type[AbstractGate]]],
    CouplingGate: Type[AbstractGate],
    periodic: bool = False,
) -> Block:
    """
    Create a brickwork block with the specified local and coupling gates.

    Args:
        wires (Sequence[Wire]): The wires to apply the gates to.
        depth (int): The depth of the brickwork block.
        LocalGates (Union[Type[AbstractGate], Sequence[Type[AbstractGate]]]): The local gates to apply to each wire.
        CouplingGate (Type[AbstractGate]): The coupling gate to apply to pairs of wires.
        periodic (bool): Whether to use periodic boundary conditions.
    Returns:
        Block: A block containing the specified brickwork structure.
    """
    block = Block()
    pairs1, pairs2 = _chunk_pairs(tuple(wires), periodic=periodic)

    if not is_bearable(LocalGates, Sequence[Type[AbstractGate]]):
        LocalGates = (LocalGates,)

    for _layer in range(depth):
        for wire in wires:
            for Gate in LocalGates:
                block.add(Gate(wires=(wire,)))
        for pairs in (pairs1, pairs2):
            for pair in pairs:
                block.add(CouplingGate(wires=pair))

    return block


@beartype
def brickwork_type(
    wires: Sequence[Wire],
    depth: int,
    ansatz: Literal["hea", "rxx", "rzz"],
    periodic: bool = False,
):
    """
    Create a brickwork block with the specified ansatz type.
    Ansatz can be one of 'hea', 'rxx', or 'rzz'.
    - 'hea' uses RX and RY gates for one-qubit gates and CZ for two-qubit gates.
    - 'rxx' uses RX and RY gates for one-qubit gates and RXX for two-qubit gates.
    - 'rzz' uses RZ gates for one-qubit gates and RZZ for two-qubit gates.

    Args:
        wires (Sequence[Wire]): The wires to apply the gates to.
        depth (int): The depth of the brickwork block.
        ansatz (Literal['hea', 'rxx', 'rzz']): The type of ansatz to use.
        periodic (bool): Whether to use periodic boundary conditions.

    Returns:
        Block: A block containing the specified brickwork ansatz.
    """
    match ansatz:
        case "hea":
            return brickwork(
                wires=wires,
                depth=depth,
                one_qubit_gates=[dv.RXGate, dv.RYGate],
                two_qubit_gates=dv.CZGate,
                periodic=periodic,
            )
        case "rxx":
            return brickwork(
                wires=wires,
                depth=depth,
                LocalGates=(dv.RXGate, dv.RYGate),
                CouplingGate=dv.RXXGate,
                periodic=periodic,
            )
        case "rzz":
            return brickwork(
                wires=wires,
                depth=depth,
                LocalGates=(dv.RXGate, dv.RYGate),
                CouplingGate=dv.RZZGate,
                periodic=periodic,
            )
