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
import copy
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import jaxtyping
import paramax
from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Sequence
from jaxtyping import ArrayLike

from squint.ops.base import (
    AbstractGate,
    AbstractMixedState,
    AbstractPureState,
    Wire,
    bases,
    create,
    destroy,
)
from squint.ops.math import (
    compile_Aij_indices,
    compute_transition_amplitudes,
    get_fixed_sum_tuples,
)

__all__ = [
    "FockState",
    "FixedEnergyFockState",
    "TwoModeWeakThermalState",
    "BeamSplitter",
    "Phase",
    "LinearOpticalUnitaryGate",
]


class FockState(AbstractPureState):
    r"""
    Fock state.
    """

    n: Sequence[tuple[complex, Sequence[int]]]

    @beartype
    def __init__(
        self,
        wires: Sequence[Wire],
        n: Sequence[int] | Sequence[tuple[complex | float, Sequence[int]]] = None,
    ):
        super().__init__(wires=wires)
        if n is None:
            n = [(1.0, (0,) * len(wires))]  # initialize to |vac> = |0, 0, ...> state
        elif is_bearable(n, Sequence[int]):
            n = [(1.0, n)]
        elif is_bearable(n, Sequence[tuple[complex | float, Sequence[int]]]):
            norm = jnp.sum(jnp.abs(jnp.array([amp for amp, wires in n])) ** 2)
            n = [(amp / jnp.sqrt(norm).item(), wires) for amp, wires in n]
        self.n = paramax.non_trainable(n)
        return
        
    def __call__(self):
        return sum(
            [
                jnp.zeros(
                    shape=[wire.dim for wire in self.wires]
                )
                .at[*term[1]]
                .set(term[0])
                for term in self.n
            ]
        )    


class FixedEnergyFockState(AbstractPureState):
    r"""
    Fixed energy Fock superposition.
    """

    weights: ArrayLike
    phases: ArrayLike
    n: int
    bases: Sequence[tuple[complex | float, Sequence[int]]]

    @beartype
    def __init__(
        self,
        wires: Sequence[Wire],
        n: int = 1,
        weights: Optional[ArrayLike] = None,
        phases: Optional[ArrayLike] = None,
        key: Optional[jaxtyping.PRNGKeyArray] = None,
    ):
        super().__init__(wires=wires)

        def fixed_energy_states(length, energy):
            if length == 1:
                yield (energy,)
            else:
                for value in range(energy + 1):
                    for permutation in fixed_energy_states(length - 1, energy - value):
                        yield (value,) + permutation

        self.n = n
        self.bases = list(fixed_energy_states(len(wires), n))
        if not weights:
            weights = jnp.ones(shape=(len(self.bases),))
            # weights = jnp.linspace(1.0, 2.0, len(self.bases))
        if not phases:
            phases = jnp.zeros(shape=(len(self.bases),))
            # phases = jnp.linspace(1.0, 2.0, len(self.bases))

        if key is not None:
            subkeys = jr.split(key, 2)
            weights = jr.normal(subkeys[0], shape=weights.shape)
            phases = jr.normal(subkeys[1], shape=phases.shape)

        self.weights = weights
        self.phases = phases
        return

    def __call__(self, dim: int):
        return jnp.einsum(
            "i, i... -> ...",
            jnp.exp(1j * self.phases) * jnp.sqrt(jax.nn.softmax(self.weights)),
            jnp.array(
                [
                    jnp.zeros(shape=(dim,) * len(self.wires)).at[*basis].set(1.0)
                    for basis in self.bases
                ]
            ),
        )


class TwoModeWeakThermalState(AbstractMixedState):
    r"""
    Two-mode weak coherent source.
    """

    g: ArrayLike
    phi: ArrayLike
    epsilon: ArrayLike

    @beartype
    def __init__(
        self,
        wires: Sequence[Wire],
        epsilon: float,
        g: float,
        phi: float,
    ):
        super().__init__(wires=wires)
        self.epsilon = jnp.array(epsilon)
        self.g = jnp.array(g)
        self.phi = jnp.array(phi)
        return

    def __call__(self):
        assert len(self.wires) == 2, "not correct wires"
        # assert dim == 2, "not correct dim"
        dims = (self.wires[0].dim, self.wires[1].dim, self.wires[0].dim, self.wires[1].dim, )
        rho = jnp.zeros(shape=dims, dtype=jnp.complex128)
        rho = rho.at[0, 0, 0, 0].set(1 - self.epsilon)
        rho = rho.at[0, 1, 0, 1].set(self.epsilon / 2)
        rho = rho.at[1, 0, 1, 0].set(self.epsilon / 2)
        rho = rho.at[0, 1, 1, 0].set(self.g * jnp.exp(1j * self.phi) * self.epsilon / 2)
        rho = rho.at[1, 0, 0, 1].set(
            self.g * jnp.exp(-1j * self.phi) * self.epsilon / 2
        )
        return rho


class TwoModeSqueezingGate(AbstractGate):
    r"""
    TwoModeSqueezingGate
    """

    r: ArrayLike
    phi: ArrayLike

    @beartype
    def __init__(self, wires: tuple[Wire, Wire], r, phi):
        super().__init__(wires=wires)
        self.r = jnp.asarray(r)
        self.phi = jnp.asarray(phi)
        return

    def __call__(self):
        dims = (self.wires[0].dim, self.wires[1].dim, self.wires[0].dim, self.wires[1].dim, )
        
        s2_l = jnp.kron(create(self.wires[0].dim), create(self.wires[1].dim))
        s2_r = jnp.kron(destroy(self.wires[0].dim), destroy(self.wires[1].dim))
        u = jax.scipy.linalg.expm(
            1j * jnp.tanh(self.r) * (jnp.conjugate(self.phi) * s2_l - self.phi * s2_r)
        ).reshape(dims)
        return u
        # return einops.rearrange(u, "a b c d -> a c b d")


class BeamSplitter(AbstractGate):
    r"""
    Beam splitter
    """

    r: ArrayLike

    @beartype
    def __init__(
        self,
        wires: tuple[Wire, Wire],
        r: float | ArrayLike = jnp.pi / 4,
    ):
        super().__init__(wires=wires)
        self.r = jnp.array(r)
        return

    def __call__(self):
        dims = (self.wires[0].dim, self.wires[1].dim, self.wires[0].dim, self.wires[1].dim, )
        
        bs_l = jnp.kron(create(self.wires[0].dim), destroy(self.wires[1].dim))
        bs_r = jnp.kron(destroy(self.wires[0].dim), create(self.wires[1].dim))
        u = jax.scipy.linalg.expm(1j * self.r * (bs_l + bs_r)).reshape(dims)
        return u  # TODO: this is correct for the `mixed` backend, while... DONE: this should be correct for both backends now
        # return einops.rearrange(u, "a b c d -> a c b d")   # TODO this is correct for the `pure`


class LinearOpticalUnitaryGate(AbstractGate):
    r""" """

    unitary_modes: ArrayLike  # unitary which acts on the optical modes

    @beartype
    def __init__(
        self,
        wires: tuple[Wire, ...],
        unitary_modes: ArrayLike,
    ):
        super().__init__(wires=wires)
        assert unitary_modes.shape == (len(wires), len(wires)), (
            "Number of wires does not match mode unitary shape."
        )
        assert jnp.allclose(
            unitary_modes @ unitary_modes.T.conj(), jnp.eye(len(wires), len(wires))
        ), "`unitary_modes` arg is not unitary."
        self.unitary_modes = unitary_modes
        return

    def _init_static_arrays(self, dim: int):
        """
        For each number of photons, n > 0, we generate all combinations and compute the A_ij square matrices.
        Next, we compute the U_{i,j} coefficients for input i and output j bases by taking the permanent.
        We insert these coefficients into the larger U_{i,j} matrix, which is a square matrix of size cut^m x cut^m.
        We need to do this for all 0 < n <= cut, where cut is the cutoff for the number of photons in each mode.
        """
        # create the indices for input and output as an ndarray. [m, dim, dim, dim, ..., dim]; we use this for the factorial and for referencing the index ordering
        m = len(self.wires)

        # generate the indices for calculating all of the factorial normalization arrays
        idx_i = jnp.indices((dim,) * m)
        idx_j = copy.deepcopy(idx_i)

        idx_i_fac = jnp.prod(jax.scipy.special.factorial(idx_i), axis=0)
        idx_j_fac = jnp.prod(jax.scipy.special.factorial(idx_j), axis=0)

        factorial_weight = jnp.einsum(
            "i,j->ij",
            1 / jnp.sqrt(idx_i_fac).reshape(dim**m),
            1 / jnp.sqrt(idx_j_fac).reshape(dim**m),
        ).reshape((dim,) * m * 2)

        # for each n <= cut, generate all combinations of indices
        # this is done for each n (number of excitations), rather than all possible number bases at once,
        # as the Aij matrix is not square, and the interferometer is by definition linear
        inds_n_i = [list(get_fixed_sum_tuples(m, n)) for n in range(dim)]
        inds_n_j = copy.deepcopy(inds_n_i)

        def pairwise_combinations(A, B):
            return jnp.stack(
                [
                    A[:, None, :].repeat(B.shape[0], axis=1),
                    B[None, :, :].repeat(A.shape[0], axis=0),
                ],
                axis=2,
            ).reshape(-1, 2, A.shape[1])

        # calculate all pairs of input & output bases, along with their transition indices for creating the Aij matrices
        pairs, transition_inds = [], []
        for n in range(dim):
            p = pairwise_combinations(jnp.array(inds_n_i[n]), jnp.array(inds_n_j[n]))
            pairs.append(p.reshape(p.shape[0], -1))

            t_inds = compile_Aij_indices(inds_n_i[n], inds_n_j[n], m, n)
            transition_inds.append(t_inds)

        return transition_inds, pairs, factorial_weight

    def __call__(self):
        # generate all of the static arrays for the indices, transition indices to create Aij for all n
        # and the factorial normalization array
        dim = self.wires[0].dim  # TODO: use the dims for all wires
        transition_inds, pairs, factorial_weight = self._init_static_arrays(dim)

        # map the unitary acting on the modes (m x m) to the unitary acting on number states,
        # computed as the Perm[Aij] for all combinations of i and j number bases
        def map_unitary(unitary_modes):
            unitary_number = jnp.zeros((dim,) * 2 * len(self.wires), dtype=jnp.complex_)
            for n in range(dim):
                coefficients = compute_transition_amplitudes(
                    unitary_modes, transition_inds[n]
                )
                unitary_number = unitary_number.at[tuple(pairs[n].T)].set(
                    coefficients.flatten()
                )
            unitary_number = unitary_number * factorial_weight
            return unitary_number

        unitary_number = map_unitary(self.unitary_modes)

        return unitary_number


# class LinearOpticalUnitaryGate(AbstractGate):
#     r"""
#     Linear optical passive element.
#     """

#     rs: ArrayLike

#     @beartype
#     def __init__(
#         self,
#         wires: tuple[Wire, ...],
#         rs: Optional[ArrayLike] = None,
#         key: Optional[jaxtyping.PRNGKeyArray] = None,
#     ):
#         super().__init__(wires=wires)
#         if rs is None:
#             rs = jnp.ones(shape=[len(wires) * (len(wires) - 1) // 2]) * jnp.pi / 4

#         if key is not None:
#             rs = jr.normal(key, shape=rs.shape)

#         self.rs = jnp.array(rs)

#     def __call__(self, dim: int):
#         combs = list(itertools.combinations(range(len(self.wires)), 2))

# _h = sum(
#     [
#         functools.reduce(
#             jnp.kron,
#             [
#                 {i: r * create(dim), j: destroy(dim)}.get(k, eye(dim))
#                 for k in range(len(self.wires))
#             ],
#         )
#         for r, (i, j) in zip(self.rs, combs, strict=False)
#     ]
#     + [
#         functools.reduce(
#             jnp.kron,
#             [
#                 {j: r.conj() * create(dim), i: destroy(dim)}.get(k, eye(dim))
#                 for k in range(len(self.wires))
#             ],
#         )
#         for r, (i, j) in zip(self.rs, combs, strict=False)
#     ]
# )
# _s_matrix = (
#     f"({' '.join([get_symbol(2 * k) for k in range(len(self.wires))])}) "
#     f"({' '.join([get_symbol(2 * k + 1) for k in range(len(self.wires))])})"
# )

# _s_tensor = (
#     " ".join([get_symbol(2 * k) for k in range(len(self.wires))])
#     + " "
#     + " ".join([get_symbol(2 * k + 1) for k in range(len(self.wires))])
# )

# dims = {get_symbol(k): dim for k in range(2 * len(self.wires))}

# u = einops.rearrange(
#     jax.scipy.linalg.expm(-1j * _h), f"{_s_matrix} -> {_s_tensor}", **dims
# )

# return u


# class QFT(AbstractGate):
#     coeff: float

#     @beartype
#     def __init__(self, wires: tuple[int, ...], coeff: float = 0.3):
#         super().__init__(wires=wires)
#         self.coeff = jnp.array(coeff)
#         return

#     def __call__(self, dim: int):
#         wires = self.wires
#         perms = list(
#             itertools.permutations(
#                 [create(dim), destroy(dim)] + [eye(dim) for _ in range(len(wires) - 2)]
#             )
#         )
#         coeff = self.coeff

#         terms = sum([functools.reduce(jnp.kron, perm) for perm in perms])
#         # u = jax.scipy.linalg.expm(1j * jnp.pi / len(wires) / 2 * terms)
#         u = jax.scipy.linalg.expm(1j * coeff * terms)

#         subscript = f"{' '.join(characters[: 2 * len(wires)])} -> {' '.join([c for i in range(len(wires)) for c in (characters[i], characters[i + len(wires)])])}"
#         logger.info(f"Subscript for QFT {subscript}")
#         logger.info(f"Number of terms {len(perms)}")
#         # return einops.rearrange(u.reshape(2 * len(wires) * (dim,)), "a b c d e f -> a d b e c f")
#         return einops.rearrange(u.reshape(2 * len(wires) * (dim,)), subscript)

# return u
# return einops.rearrange(u, subscript)


class Phase(AbstractGate):
    r"""
    Phase gate.
    """

    phi: ArrayLike

    @beartype
    def __init__(
        self,
        wires: tuple[Wire] = (0,),
        phi: float | int | ArrayLike = 0.0,
    ):
        super().__init__(wires=wires)
        self.phi = jnp.array(phi)
        return

    def __call__(self):
        return jnp.diag(jnp.exp(1j * bases(self.wires[0].dim) * self.phi))


fock_subtypes = {FockState, BeamSplitter, Phase}

# %%
if __name__ == "__main__":
    # %%
    from squint.utils import print_nonzero_entries

    dim = 3
    wires = (0, 1, 2)
    U = 0.5 * jnp.array(
        [
            # [1.0, -1.0],
            # [-1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    )
    op = LinearOpticalUnitaryGate(wires=wires, unitary_modes=U)

    @jax.jit
    def f():
        return op(dim)

    print_nonzero_entries(f())
# %%
