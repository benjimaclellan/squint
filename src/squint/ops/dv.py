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
import math
from typing import Union, Callable

import jax.numpy as jnp
import jax.scipy as jsp
import paramax
from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Sequence, Type
from jaxtyping import ArrayLike, Float, Inexact, Scalar

from squint.ops.base import (
    AbstractGate,
    AbstractMixedState,
    AbstractPureState,
    Wire,
    bases,
    basis_operators,
)

__all__ = [
    "DiscreteVariableState",
    "MaximallyMixedState",
    "XGate",
    "ZGate",
    "HGate",
    "RZGate",
    "RYGate",
    "RXGate",
    "RXXGate",
    "Conditional",
    "TwoLocalHermitianBasisGate",
]


class DiscreteVariableState(AbstractPureState):
    r"""
    A pure quantum state for a discrete variable system.

    $|\psi\rangle = \sum_{i} a_i |i\rangle$ where $a_i$ are amplitudes and $|i\rangle$ are basis states.
    """

    n: Sequence[
        tuple[complex, Sequence[int]]
    ]  # todo: add superposition as n, using second typehint

    @beartype
    def __init__(
        self,
        wires: Sequence[Wire],
        n: Sequence[int] | Sequence[tuple[complex | float, Sequence[int]]] = None,
    ):
        super().__init__(wires=wires)
        if n is None:
            n = [(1.0, (0,) * len(wires))]  # initialize to |0, 0, ...> state
        elif is_bearable(n, Sequence[int]):
            n = [(1.0, n)]
        elif is_bearable(n, Sequence[tuple[complex | float, Sequence[int]]]):
            norm = jnp.sum(jnp.abs(jnp.array([i[0] for i in n])) ** 2)
            n = [((amp / jnp.sqrt(norm)).item(), basis) for amp, basis in n]
        self.n = paramax.non_trainable(n)
        return

    def __call__(self):
        return sum(
            [
                jnp.zeros(
                    # shape=(dim,) * len(self.wires)
                    shape=[wire.dim for wire in self.wires]
                )
                .at[*term[1]]
                .set(term[0])
                for term in self.n
            ]
        )


class MaximallyMixedState(AbstractMixedState):
    r"""
    The maximally mixed state for discrete variable systems.

    Represents the completely mixed density matrix $\rho = I/d$ where $d$ is the
    total Hilbert space dimension. This state has maximum von Neumann entropy
    and represents complete ignorance about the quantum state.

    The density matrix is constructed as:
    $$\rho = \frac{1}{d} \sum_{i=0}^{d-1} |i\rangle\langle i|$$

    where $d = \prod_i d_i$ is the product of all wire dimensions.

    Example:
        ```python
        wire = Wire(dim=2, idx=0)
        state = MaximallyMixedState(wires=(wire,))
        # Creates rho = [[0.5, 0], [0, 0.5]]
        ```

    Note:
        This state requires the "mixed" backend in the circuit.
    """

    @beartype
    def __init__(
        self,
        wires: Sequence[Wire],
    ):
        super().__init__(wires=wires)

    def __call__(self):
        dims = [wire.dim for wire in self.wires]
        d = math.prod(dims)
        identity = jnp.eye(d, dtype=jnp.complex128) / d
        tensor = identity.reshape(tuple(dim for dim in dims for _ in range(2)))
        return tensor


def x(dim):
    return jnp.roll(jnp.eye(dim, k=0), shift=1, axis=0)

def z(dim):
    return jnp.diag(
            jnp.exp(1j * 2 * jnp.pi * jnp.arange(dim) / dim)
        )

def eye(dim):
    return jnp.eye(dim)


class XGate(AbstractGate):
    r"""
    The generalized shift operator, which when `dim = 2` corresponds to the standard $X$ gate.

    $U = \sum_{k=0}^{d-1} |k\rangle \langle (k+1) \mod d|$
    """

    @beartype
    def __init__(
        self,
        wires: tuple[Wire] = (0,),
    ):
        super().__init__(wires=wires)
        return

    def __call__(self):
        return x(self.wires[0].dim)


class ZGate(AbstractGate):
    r"""
    The generalized phase operator, which when `dim = 2` corresponds to the standard $Z$ gate.

    $U = \sum_{k=0}^{d-1} e^{2\pi i k / d} |k\rangle\langle k|$
    """

    @beartype
    def __init__(
        self,
        wires: tuple[Wire] = (0,),
    ):
        super().__init__(wires=wires)
        return

    def __call__(self):
        return z(self.wires[0].dim)
        # return jnp.diag(
            # jnp.exp(1j * 2 * jnp.pi * jnp.arange(self.wires[0].dim) / self.wires[0].dim)
        # )


class HGate(AbstractGate):
    r"""
    The generalized discrete Fourier operator, which when `dim = 2` corresponds to the standard $H$ gate.

    $U = \frac{1}{\sqrt{d}} \sum_{j,k=0}^{d-1} e^{2\pi i jk / d} |j\rangle\langle k|$
    """

    @beartype
    def __init__(
        self,
        wires: tuple[Wire] = (0,),
    ):
        super().__init__(wires=wires)
        return

    def __call__(self):
        dim = self.wires[0].dim
        return jnp.exp(
            1j
            * 2
            * jnp.pi
            / dim
            * jnp.einsum("a,b->ab", jnp.arange(dim), jnp.arange(dim))
        ) / jnp.sqrt(dim)


class Conditional(AbstractGate):
    r"""
    The generalized conditional operator.
    Applies gate $U$ raised to a power conditional on the control state:
    $U = \sum_{k=0}^{d-1} |k\rangle\langle k| \otimes U^k$
    """

    # gate: Union[XGate, ZGate]  # type: ignore
    ufunc: Callable
    
    @beartype
    def __init__(
        self,
        # gate: Union[Type[XGate], Type[ZGate]],
        ufunc: Callable = eye,
        wires: tuple[Wire, Wire] = (0, 1),
    ):
        super().__init__(wires=wires)
        self.ufunc = ufunc
        # self.gate = gate(wires=(wires[1],))
        return

    def __call__(self):
        u = sum(
            [
                jnp.einsum(
                    "ac,bd -> abcd",
                    jnp.zeros(shape=(self.wires[0].dim, self.wires[0].dim))
                    .at[i, i]
                    .set(1.0),
                    # jnp.linalg.matrix_power(self.gate(), i),
                    jnp.linalg.matrix_power(self.ufunc(self.wires[1].dim), i),
                )
                for i in range(self.wires[0].dim)
            ]
        )

        return u


class CXGate(Conditional):
    r"""
    Controlled-X (CNOT) gate for qubits and qudits.

    Applies an X gate to the target qubit/qudit conditional on the control.
    For qubits (dim=2), this is the standard CNOT gate:
    $$CX = |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes X$$

    For qudits (dim>2), this generalizes to:
    $$CX = \sum_{k=0}^{d-1} |k\rangle\langle k| \otimes X^k$$

    Args:
        wires: Tuple of (control_wire, target_wire).

    Example:
        ```python
        control = Wire(dim=2, idx=0)
        target = Wire(dim=2, idx=1)
        cx = CXGate(wires=(control, target))
        ```
    """

    @beartype
    def __init__(
        self,
        wires: tuple[Wire, Wire] = (0, 1),
    ):
        super().__init__(wires=wires, ufunc=x)


class CZGate(Conditional):
    r"""
    Controlled-Z gate for qubits and qudits.

    Applies a Z gate to the target qubit/qudit conditional on the control.
    For qubits (dim=2), this is the standard CZ gate:
    $$CZ = |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes Z$$

    For qudits (dim>2), this generalizes to:
    $$CZ = \sum_{k=0}^{d-1} |k\rangle\langle k| \otimes Z^k$$

    Args:
        wires: Tuple of (control_wire, target_wire).

    Example:
        ```python
        control = Wire(dim=2, idx=0)
        target = Wire(dim=2, idx=1)
        cz = CZGate(wires=(control, target))
        ```
    """

    @beartype
    def __init__(
        self,
        wires: tuple[Wire, Wire] = (0, 1),
    ):
        super().__init__(wires=wires, ufunc=z)


class EmbeddedRGate(AbstractGate):
    theta: ArrayLike
    phi: ArrayLike
    levels: tuple[int, int]

    @beartype
    def __init__(
        self,
        wires: tuple[Wire] = (0,),
        levels: tuple[int, int] = (0, 1),
        theta: float | int = 0.0,
        phi: float | int = 0.0,
    ):
        super().__init__(wires=wires)
        self.theta = jnp.array(theta)
        self.phi = jnp.array(phi)
        self.levels = levels
        return

    def __call__(self):
        dim = self.wires[0].dim
        level_a = jnp.zeros(dim).at[self.levels[0]].set(1.0)
        level_b = jnp.zeros(dim).at[self.levels[1]].set(1.0)
        inds = tuple(
            [
                jnp.array([i for i in range(dim) if i not in self.levels]),
                jnp.array([i for i in range(dim) if i not in self.levels]),
            ]
        )
        return (
            jnp.zeros((dim, dim), dtype=jnp.complex128).at[inds].set(1.0)
            + jnp.cos(self.theta / 2) * jnp.einsum("i,j->ij", level_a, level_a)
            + jnp.cos(self.theta / 2) * jnp.einsum("i,j->ij", level_b, level_b)
            - 1j
            * jnp.exp(-1j * self.phi)
            * jnp.sin(self.theta / 2)
            * jnp.einsum("i,j->ij", level_a, level_b)
            - 1j
            * jnp.exp(1j * self.phi)
            * jnp.sin(self.theta / 2)
            * jnp.einsum("i,j->ij", level_b, level_a)
        )


class RZGate(AbstractGate):
    r"""
    Rotation gate around the Z-axis for qubits and qudits.

    For qubits (dim=2), this implements the standard RZ rotation:
    $$R_Z(\phi) = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\phi} \end{pmatrix}$$

    For qudits (dim>2), this generalizes to:
    $$R_Z(\phi) = \sum_{k=0}^{d-1} e^{ik\phi} |k\rangle\langle k|$$

    Attributes:
        phi (ArrayLike): The rotation angle in radians.

    Example:
        ```python
        wire = Wire(dim=2, idx=0)
        rz = RZGate(wires=(wire,), phi=jnp.pi/4)
        ```
    """

    phi: ArrayLike

    @beartype
    def __init__(
        self,
        wires: tuple[Wire] = (0,),
        # phi: float | int = 0.0,
        phi: float | int | Float[Scalar, ""] = 0.0
    ):
        super().__init__(wires=wires)
        self.phi = jnp.array(phi)
        return

    def __call__(self):
        return jnp.diag(jnp.exp(1j * bases(self.wires[0].dim) * self.phi))


class RXGate(AbstractGate):
    r"""
    Rotation gate around the X-axis for qubits.

    Implements the standard RX rotation:
    $$R_X(\phi) = \cos(\phi/2) I - i \sin(\phi/2) X = \begin{pmatrix} \cos(\phi/2) & -i\sin(\phi/2) \\ -i\sin(\phi/2) & \cos(\phi/2) \end{pmatrix}$$

    Attributes:
        phi (ArrayLike): The rotation angle in radians.

    Note:
        This gate is only defined for qubits (dim=2).

    Example:
        ```python
        wire = Wire(dim=2, idx=0)
        rx = RXGate(wires=(wire,), phi=jnp.pi/2)
        ```
    """

    phi: ArrayLike

    @beartype
    def __init__(
        self,
        wires: tuple[Wire] = (0,),
        phi: float | int = 0.0,
        # phi: Inexact[Scalar] = 0.0
    ):
        assert wires[0].dim == 2, "RXGate only defined for dim=2."
        super().__init__(wires=wires)
        self.phi = jnp.array(phi)
        return

    def __call__(self):
        return (
            jnp.cos(self.phi / 2) * basis_operators(self.wires[0].dim)[3]  # identity
            - 1j * jnp.sin(self.phi / 2) * basis_operators(self.wires[0].dim)[2]  # X
        )


class RYGate(AbstractGate):
    r"""
    Rotation gate around the Y-axis for qubits.

    Implements the standard RY rotation:
    $$R_Y(\phi) = \cos(\phi/2) I - i \sin(\phi/2) Y = \begin{pmatrix} \cos(\phi/2) & -\sin(\phi/2) \\ \sin(\phi/2) & \cos(\phi/2) \end{pmatrix}$$

    Attributes:
        phi (ArrayLike): The rotation angle in radians.

    Note:
        This gate is only defined for qubits (dim=2).

    Example:
        ```python
        wire = Wire(dim=2, idx=0)
        ry = RYGate(wires=(wire,), phi=jnp.pi/2)
        ```
    """

    phi: ArrayLike

    @beartype
    def __init__(
        self,
        wires: tuple[Wire] = (0,),
        phi: float | int = 0.0,
    ):
        assert wires[0].dim == 2, "RYGate only defined for dim=2."

        super().__init__(wires=wires)
        self.phi = jnp.array(phi)
        return

    def __call__(self):
        return (
            jnp.cos(self.phi / 2) * basis_operators(self.wires[0].dim)[3]  # identity
            - 1j * jnp.sin(self.phi / 2) * basis_operators(self.wires[0].dim)[1]  # Y
        )


# class CholeskyDecompositionGate(AbstractGate):
#     decomp: ArrayLike  # lower triangular matrix for Cholesky decomposition of hermitian matrix

#     _dim: int
#     _subscripts: str

#     @beartype
#     def __init__(
#         self,
#         wires: tuple[Wire, ...],
#         dim: int,
#         key: Optional[PRNGKeyArray] = None,
#     ):
#         super().__init__(wires=wires)
#         if not key:
#             key = jr.PRNGKey(time.time_ns())
#         # self.decomp = jnp.ones(shape=(dim ** len(wires), dim ** len(wires)), dtype=jnp.complex_)  # todo
#         self.decomp = jr.normal(
#             key=key, shape=(dim ** len(wires), dim ** len(wires)), dtype=jnp.complex_
#         )  # todo
#         self._dim = dim
#         self._subscripts = f"{' '.join(characters[0 : 2 * len(wires)])} -> {' '.join(characters[0 : 2 * len(wires) : 2])} {' '.join(characters[1 : 2 * len(wires) : 2])}"
#         return

#     def __call__(self, dim: int):
#         tril = jnp.tril(self.decomp)
#         herm = tril @ tril.conj().T
#         u = jsp.linalg.expm(1j * herm).reshape((2 * len(self.wires)) * (dim,))
#         return einops.rearrange(u, self._subscripts)  # todo: interleave reshaping


class TwoLocalHermitianBasisGate(AbstractGate):
    r"""
    Two-qubit/qudit gate generated by a tensor product of Gell-Mann basis operators.

    Implements gates of the form:
    $$U(\theta) = \exp(-i \theta \cdot G_i \otimes G_j)$$

    where $G_i$ and $G_j$ are Gell-Mann basis operators (generalized Pauli matrices)
    acting on the first and second wire respectively. For qubits (dim=2), the
    Gell-Mann operators reduce to the Pauli matrices.

    This is the base class for specific two-qubit interaction gates like RXXGate
    and RZZGate.

    Attributes:
        angles (ArrayLike): The rotation angle(s) in radians.
        _basis_op_indices (tuple[int, int]): Indices of the Gell-Mann basis operators
            to use on each wire. For dim=2: 0=Z, 1=Y, 2=X, 3=I.

    Example:
        ```python
        wire0 = Wire(dim=2, idx=0)
        wire1 = Wire(dim=2, idx=1)
        # Create an XX interaction gate
        gate = TwoLocalHermitianBasisGate(
            wires=(wire0, wire1),
            angles=jnp.pi/4,
            _basis_op_indices=(2, 2)  # X tensor X
        )
        ```
    """

    angles: ArrayLike
    _basis_op_indices: tuple[
        int, int
    ]  # index of basis (Gell-Mann) ops to apply on the first and second wires, respectively

    @beartype
    def __init__(
        self,
        wires: tuple[Wire, Wire],
        angles: Union[float, int, Sequence[int], Sequence[float], ArrayLike],
        _basis_op_indices: tuple[int, int] = (2, 2),
    ):
        super().__init__(wires=wires)

        self.angles = jnp.array(angles)
        self._basis_op_indices = _basis_op_indices
        return

    def _hermitian_op(self):
        return jnp.kron(
            basis_operators(self.wires[0].dim)[self._basis_op_indices[0]],
            basis_operators(self.wires[1].dim)[self._basis_op_indices[1]],
        )

    def _rearrange(self, tensor: ArrayLike):
        return tensor.reshape(
            self.wires[0].dim,
            self.wires[1].dim,
            self.wires[0].dim,
            self.wires[1].dim,
        )

    # def _dim_check(self, dim: int):
    # raise NotImplementedError()

    def __call__(self):
        # return self._rearrange(self._hermitian_op(dim), dim)
        # return self._hermitian_op(dim)
        # self._dim_check(dim)
        return self._rearrange(
            jsp.linalg.expm(-1j * self.angles * self._hermitian_op())
        )


class RXXGate(TwoLocalHermitianBasisGate):
    @beartype
    def __init__(
        self,
        wires: tuple[Wire, Wire],
        angle: Union[float, int, Float[ArrayLike, "..."]] = 0.0,  # TODO: initialize
    ):
        # PauliX is index 2 for dim=2
        assert wires[0].dim == 2 and wires[1].dim == 2, (
            "RXXGate can only be applied when dim=2."
        )

        super().__init__(wires=wires, angles=jnp.array(angle), _basis_op_indices=(2, 2))
        return


class RZZGate(TwoLocalHermitianBasisGate):
    @beartype
    def __init__(
        self,
        wires: tuple[Wire, Wire],
        angle: Union[float, int, Float[ArrayLike, "..."]] = 0.0,  # TODO: initialize
    ):
        assert wires[0].dim == 2 and wires[1].dim == 2, (
            "RZZGate can only be applied when dim=2."
        )

        # PauliZ is index 0 for dim=2
        super().__init__(wires=wires, angles=jnp.array(angle), _basis_op_indices=(0, 0))
        return


# dv_subtypes = {DiscreteVariableState, XGate, ZGate, HGate, Conditional, RZGate}
