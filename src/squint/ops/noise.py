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
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Sequence
from jaxtyping import ArrayLike
from opt_einsum.parser import get_symbol

from squint.ops.base import (
    AbstractErasureChannel,
    AbstractKrausChannel,
    Wire,
    basis_operators,
)


class ErasureChannel(AbstractErasureChannel):
    r"""
    Erasure channel that traces out specified wires.

    This channel performs a partial trace over the specified wires, effectively
    "erasing" those subsystems from the quantum state. It models complete photon
    loss or the discarding of quantum information in those modes.

    Mathematically, for a density matrix $\rho$ on wires A and B, tracing out B gives:
    $$\text{Tr}_B[\rho] = \sum_i \langle i_B | \rho | i_B \rangle$$

    Note:
        This channel requires the "mixed" backend in the circuit, as it produces
        a reduced density matrix.

    Example:
        ```python
        wire0 = Wire(dim=2, idx=0)
        wire1 = Wire(dim=2, idx=1)
        circuit = Circuit(backend="mixed")
        # ... add states and operations ...
        circuit.add(ErasureChannel(wires=(wire1,)))  # Trace out wire1
        ```
    """

    @beartype
    def __init__(self, wires: Sequence[Wire]):
        super().__init__(wires=wires)
        return

    def __call__(self):
        subscripts = [
            get_symbol(2 * i) + get_symbol(2 * i + 1) for i in range(len(self.wires))
        ]
        return jnp.einsum(
            f"{','.join(subscripts)} -> {''.join(subscripts)}",
            *(jnp.identity(wire.dim) for wire in self.wires),
        )


class BitFlipChannel(AbstractKrausChannel):
    r"""
    Qubit bit flip channel.

    Models random bit flip errors with probability $p$. The channel flips
    the qubit state $|0\rangle \leftrightarrow |1\rangle$ with probability $p$
    and leaves it unchanged with probability $1-p$.

    Kraus operators:
    $$K_0 = \sqrt{1-p} \cdot I, \quad K_1 = \sqrt{p} \cdot X$$

    The channel acts on a density matrix as:
    $$\mathcal{E}(\rho) = (1-p)\rho + p X\rho X$$

    Attributes:
        p (ArrayLike): Bit flip probability, must be in [0, 1].

    Note:
        This channel is only defined for qubits (dim=2) and requires
        the "mixed" backend.

    Example:
        ```python
        wire = Wire(dim=2, idx=0)
        circuit = Circuit(backend="mixed")
        # ... add state ...
        circuit.add(BitFlipChannel(wires=(wire,), p=0.1))  # 10% bit flip probability
        ```
    """

    p: ArrayLike

    @beartype
    def __init__(self, wires: tuple[Wire], p: float):
        assert wires[0].dim == 2, "BitFlipChannel only valid for dim=2"

        super().__init__(wires=wires)
        self.p = jnp.array(p)
        # self.p = p  #paramax.non_trainable(p)
        return

    def __call__(self):
        return jnp.array(
            [
                jnp.sqrt(1 - self.p)
                * basis_operators(self.wires[0].dim)[3],  # identity
                jnp.sqrt(self.p) * basis_operators(self.wires[0].dim)[2],  # X
            ]
        )


class PhaseFlipChannel(AbstractKrausChannel):
    r"""
    Qubit phase flip (dephasing) channel.

    Models random phase flip errors with probability $p$. The channel applies
    a Z gate (phase flip) with probability $p$ and leaves the state unchanged
    with probability $1-p$.

    Kraus operators:
    $$K_0 = \sqrt{1-p} \cdot I, \quad K_1 = \sqrt{p} \cdot Z$$

    The channel acts on a density matrix as:
    $$\mathcal{E}(\rho) = (1-p)\rho + p Z\rho Z$$

    This channel preserves populations but decoheres superpositions in the
    computational basis.

    Attributes:
        p (ArrayLike): Phase flip probability, must be in [0, 1].

    Note:
        This channel is only defined for qubits (dim=2) and requires
        the "mixed" backend.

    Example:
        ```python
        wire = Wire(dim=2, idx=0)
        circuit = Circuit(backend="mixed")
        # ... add state ...
        circuit.add(PhaseFlipChannel(wires=(wire,), p=0.1))  # 10% dephasing
        ```
    """

    p: ArrayLike

    @beartype
    def __init__(self, wires: tuple[Wire], p: float):
        assert wires[0].dim == 2, "PhaseFlipChannel only valid for dim=2"

        super().__init__(wires=wires)
        self.p = jnp.array(p)
        # self.p = p  #paramax.non_trainable(p)
        return

    def __call__(self):
        return jnp.array(
            [
                jnp.sqrt(1 - self.p)
                * basis_operators(self.wires[0].dim)[3],  # identity
                jnp.sqrt(self.p) * basis_operators(self.wires[0].dim)[0],  # Z
            ]
        )


class DepolarizingChannel(AbstractKrausChannel):
    r"""
    Qubit depolarizing channel.

    Models symmetric noise that randomly applies one of the three Pauli errors
    (X, Y, or Z) each with probability $p/4$, or leaves the state unchanged
    with probability $1 - 3p/4$. At $p=1$, the state is fully depolarized to
    the maximally mixed state.

    Kraus operators:
    $$K_0 = \sqrt{1-3p/4} \cdot I, \quad K_1 = \sqrt{p/4} \cdot X$$
    $$K_2 = \sqrt{p/4} \cdot Y, \quad K_3 = \sqrt{p/4} \cdot Z$$

    The channel acts on a density matrix as:
    $$\mathcal{E}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)$$

    which can also be written as:
    $$\mathcal{E}(\rho) = (1-p)\rho + p \cdot \frac{I}{2}$$

    Attributes:
        p (ArrayLike): Depolarizing probability, must be in [0, 1].
            At p=0, no noise. At p=1, output is maximally mixed.

    Note:
        This channel is only defined for qubits (dim=2) and requires
        the "mixed" backend.

    Example:
        ```python
        wire = Wire(dim=2, idx=0)
        circuit = Circuit(backend="mixed")
        # ... add state ...
        circuit.add(DepolarizingChannel(wires=(wire,), p=0.1))  # 10% depolarization
        ```
    """

    p: ArrayLike

    @beartype
    def __init__(self, wires: tuple[Wire], p: float):
        assert wires[0].dim == 2, "DepolarizingChannel only valid for dim=2"

        super().__init__(wires=wires)
        self.p = jnp.array(p)
        return

    def __call__(self):
        return jnp.array(
            [
                jnp.sqrt(1 - 3 * self.p / 4)
                * basis_operators(self.wires[0].dim)[3],  # identity
                jnp.sqrt(self.p / 4) * basis_operators(self.wires[0].dim)[0],  # Z
                jnp.sqrt(self.p / 4) * basis_operators(self.wires[0].dim)[1],  # Y
                jnp.sqrt(self.p / 4) * basis_operators(self.wires[0].dim)[2],  # X
            ]
        )


# %%
