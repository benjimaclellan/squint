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
from typing import Union

import jax.numpy as jnp
import jax.scipy as jsp
import paramax
from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Sequence, Type
from jaxtyping import ArrayLike, Float

from squint.ops.base import (
    AbstractGate,
    AbstractMixedState,
    AbstractPureState,
    WiresTypes,
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
    "dv_subtypes",
]


class DiscreteVariableState(AbstractPureState):
    r"""
    A pure quantum state for a discrete variable system.

    $\ket{\rho} \in \sum_{(a_i, \textbf{i}) \in n} a_i \ket{\textbf{i}}$
    """

    n: Sequence[
        tuple[complex, Sequence[int]]
    ]  # todo: add superposition as n, using second typehint

    @beartype
    def __init__(
        self,
        wires: Sequence[WiresTypes],
        n: Sequence[int] | Sequence[tuple[complex | float, Sequence[int]]] = None,
    ):
        super().__init__(wires=wires)
        if n is None:
            n = [(1.0, (0,) * len(wires))]  # initialize to |0, 0, ...> state
        elif is_bearable(n, Sequence[int]):
            n = [(1.0, n)]
        elif is_bearable(n, Sequence[tuple[complex | float, Sequence[int]]]):
            norm = jnp.sqrt(jnp.sum(jnp.array([i[0] for i in n]))).item()
            n = [(amp / norm, wires) for amp, wires in n]
        self.n = paramax.non_trainable(n)
        return

    def __call__(self, dim: int):
        return sum(
            [
                jnp.zeros(shape=(dim,) * len(self.wires)).at[*term[1]].set(term[0])
                for term in self.n
            ]
        )


class MaximallyMixedState(AbstractMixedState):
    r""" """

    @beartype
    def __init__(
        self,
        wires: Sequence[WiresTypes],
    ):
        super().__init__(wires=wires)

    def __call__(self, dim: int):
        d = dim ** len(self.wires)
        identity = jnp.eye(d, dtype=jnp.complex128) / d
        tensor = identity.reshape([dim] * len(self.wires) * 2)
        return tensor


class XGate(AbstractGate):
    r"""
    The generalized shift operator, which when `dim = 2` corresponds to the standard $X$ gate.

    $U = \sum_{k=1}^d \ket{k} \bra{(k+1) \text{mod} d}$
    """

    @beartype
    def __init__(
        self,
        wires: tuple[WiresTypes] = (0,),
    ):
        super().__init__(wires=wires)
        return

    def __call__(self, dim: int):
        return jnp.roll(jnp.eye(dim, k=0), shift=1, axis=0)


class ZGate(AbstractGate):
    r"""
    The generalized phase operator, which when `dim = 2` corresponds to the standard $Z$ gate.

    $U = \sum_{k=1}^d e^{2i \pi k /d} \ket{k}\bra{k}$
    """

    @beartype
    def __init__(
        self,
        wires: tuple[WiresTypes] = (0,),
    ):
        super().__init__(wires=wires)
        return

    def __call__(self, dim: int):
        return jnp.diag(jnp.exp(1j * 2 * jnp.pi * jnp.arange(dim) / dim))


class HGate(AbstractGate):
    r"""
    The generalized discrete Fourier operator, which when `dim = 2` corresponds to the standard $H$ gate.

    $U = \sum_{j,k=1}^d e^{2 i \pi j k  / d} \ket{j}\bra{k}$
    """

    @beartype
    def __init__(
        self,
        wires: tuple[WiresTypes] = (0,),
    ):
        super().__init__(wires=wires)
        return

    def __call__(self, dim: int):
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
    If the gate $U$ is applied conditional on the state,
    $U = \sum_{k=1}^d \ket{k} \bra{k} \otimes U^k$
    """

    gate: Union[XGate, ZGate]  # type: ignore

    @beartype
    def __init__(
        self,
        gate: Union[Type[XGate], Type[ZGate]],
        wires: tuple[WiresTypes, WiresTypes] = (0, 1),
    ):
        super().__init__(wires=wires)
        self.gate = gate(wires=(wires[1],))
        return

    def __call__(self, dim: int):
        u = sum(
            [
                jnp.einsum(
                    "ac,bd -> abcd",
                    jnp.zeros(shape=(dim, dim)).at[i, i].set(1.0),
                    jnp.linalg.matrix_power(self.gate(dim=dim), i),
                )
                for i in range(dim)
            ]
        )

        return u


class CXGate(Conditional):
    @beartype
    def __init__(
        self,
        wires: tuple[WiresTypes, WiresTypes] = (0, 1),
    ):
        super().__init__(wires=wires, gate=XGate)


class CZGate(Conditional):
    @beartype
    def __init__(
        self,
        wires: tuple[WiresTypes, WiresTypes] = (0, 1),
    ):
        super().__init__(wires=wires, gate=ZGate)


class EmbeddedRGate(AbstractGate):
    theta: ArrayLike
    phi: ArrayLike
    levels: tuple[int, int]
    
    @beartype
    def __init__(
        self,
        wires: tuple[WiresTypes] = (0,),
        levels: tuple[int, int] = (0, 1),
        theta: float | int = 0.0,
        phi: float | int = 0.0,
    ):
        super().__init__(wires=wires)
        self.theta = jnp.array(theta)
        self.phi = jnp.array(phi)
        self.levels = levels
        return

    def __call__(self, dim: int):
        level_a = jnp.zeros(dim).at[self.levels[0]].set(1.0)
        level_b = jnp.zeros(dim).at[self.levels[1]].set(1.0)        
        inds = tuple([jnp.array([i for i in range(dim) if i not in self.levels]), jnp.array([i for i in range(dim) if i not in self.levels])])
        return (
            jnp.zeros((dim, dim), dtype=jnp.complex128).at[inds].set(1.0) 
            + jnp.cos(self.theta / 2) * jnp.einsum('i,j->ij', level_a, level_a)
            + jnp.cos(self.theta / 2) * jnp.einsum('i,j->ij', level_b, level_b)
            - 1j * jnp.exp(-1j * self.phi) * jnp.sin(self.theta / 2) * jnp.einsum('i,j->ij', level_a, level_b)
            - 1j * jnp.exp(1j * self.phi) * jnp.sin(self.theta / 2) * jnp.einsum('i,j->ij', level_b, level_a)
        )


class RZGate(AbstractGate):
    phi: ArrayLike

    @beartype
    def __init__(
        self,
        wires: tuple[WiresTypes] = (0,),
        phi: float | int = 0.0,
    ):
        super().__init__(wires=wires)
        self.phi = jnp.array(phi)
        return

    def __call__(self, dim: int):
        return jnp.diag(jnp.exp(1j * bases(dim) * self.phi))


class RXGate(AbstractGate):
    r"""
    The qubit RXGate gate
    """

    phi: ArrayLike

    @beartype
    def __init__(
        self,
        wires: tuple[WiresTypes] = (0,),
        phi: float | int = 0.0,
    ):
        super().__init__(wires=wires)
        self.phi = jnp.array(phi)
        return

    def __call__(self, dim: int):
        assert dim == 2, "RXGate only for dim=2"
        return (
            jnp.cos(self.phi / 2) * basis_operators(dim=2)[3]  # identity
            - 1j * jnp.sin(self.phi / 2) * basis_operators(dim=2)[2]  # X
        )


class RYGate(AbstractGate):
    r"""
    The qubit RYGate gate
    """

    phi: ArrayLike

    @beartype
    def __init__(
        self,
        wires: tuple[WiresTypes] = (0,),
        phi: float | int = 0.0,
    ):
        super().__init__(wires=wires)
        self.phi = jnp.array(phi)
        return

    def __call__(self, dim: int):
        assert dim == 2, "RYGate only for dim=2"
        return (
            jnp.cos(self.phi / 2) * basis_operators(dim=2)[3]  # identity
            - 1j * jnp.sin(self.phi / 2) * basis_operators(dim=2)[1]  # Y
        )


# class CholeskyDecompositionGate(AbstractGate):
#     decomp: ArrayLike  # lower triangular matrix for Cholesky decomposition of hermitian matrix

#     _dim: int
#     _subscripts: str

#     @beartype
#     def __init__(
#         self,
#         wires: tuple[WiresTypes, ...],
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
    angles: ArrayLike
    _basis_op_indices: tuple[
        int, int
    ]  # index of basis (Gell-Mann) ops to apply on the first and second wires, respectively

    @beartype
    def __init__(
        self,
        wires: tuple[WiresTypes, WiresTypes],
        angles: Union[float, int, Sequence[int], Sequence[float], ArrayLike],
        _basis_op_indices: tuple[int, int] = (2, 2),
    ):
        super().__init__(wires=wires)

        self.angles = jnp.array(angles)
        self._basis_op_indices = _basis_op_indices
        return

    def _hermitian_op(self, dim: int):
        return jnp.kron(
            basis_operators(dim)[self._basis_op_indices[0]],
            basis_operators(dim)[self._basis_op_indices[1]],
        )

    def _rearrange(self, tensor: ArrayLike, dim: int):
        return tensor.reshape(4 * (dim,))
        # return einops.rearrange(tensor.reshape(4 * (dim,)), "a b c d -> a c b d")

    def _dim_check(self, dim: int):
        raise NotImplementedError()

    def __call__(self, dim: int):
        # return self._rearrange(self._hermitian_op(dim), dim)
        # return self._hermitian_op(dim)
        self._dim_check(dim)
        return self._rearrange(
            jsp.linalg.expm(-1j * self.angles * self._hermitian_op(dim)), dim
        )


class RXXGate(TwoLocalHermitianBasisGate):
    @beartype
    def __init__(
        self,
        wires: tuple[WiresTypes, WiresTypes],
        angle: Union[float, int, Float[ArrayLike, "..."]] = 0.0,  # TODO: initialize
    ):
        # PauliX is index 2 for dim=2
        super().__init__(wires=wires, angles=jnp.array(angle), _basis_op_indices=(2, 2))
        return

    def _dim_check(self, dim: int):
        assert dim == 2, "RXXGate can only be applied when dim=2."


class RZZGate(TwoLocalHermitianBasisGate):
    @beartype
    def __init__(
        self,
        wires: tuple[WiresTypes, WiresTypes],
        angle: Union[float, int, Float[ArrayLike, "..."]] = 0.0,  # TODO: initialize
    ):
        # PauliZ is index 0 for dim=2
        super().__init__(wires=wires, angles=jnp.array(angle), _basis_op_indices=(0, 0))
        return

    def _dim_check(self, dim: int):
        assert dim == 2, "RXXGate can only be applied when dim=2."


dv_subtypes = {DiscreteVariableState, XGate, ZGate, HGate, Conditional, RZGate}
