# %%
import functools
import itertools
from typing import Optional

import einops
import jax
import jax.numpy as jnp
import paramax
from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Sequence
from jaxtyping import ArrayLike
from opt_einsum import get_symbol

from squint.ops.base import (
    AbstractGate,
    AbstractMixedState,
    AbstractPureState,
    bases,
    create,
    destroy,
    eye,
)

__all__ = ["FockState", "BeamSplitter", "Phase", "S2", "QFT", "fock_subtypes"]


class FockState(AbstractPureState):
    r"""
    Fock state.
    """

    n: Sequence[tuple[complex, Sequence[int]]]

    @beartype
    def __init__(
        self,
        wires: Sequence[int],
        n: Sequence[int] | Sequence[tuple[complex | float, Sequence[int]]] = None,
    ):
        super().__init__(wires=wires)
        if n is None:
            n = [(1.0, (1,) * len(wires))]  # initialize to |1, 1, ...> state
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
        wires: Sequence[int],
        n: int = 1,
        weights: Optional[ArrayLike] = None,
        phases: Optional[ArrayLike] = None,
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
            self.weights = jnp.ones(shape=(len(self.bases),), dtype=jnp.float64)
        if not phases:
            self.phases = jnp.zeros(shape=(len(self.bases),), dtype=jnp.float64)
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


class TwoModeWeakThermalSource(AbstractMixedState):
    r"""
    Two-mode weak coherent source.
    """

    g: ArrayLike
    phi: ArrayLike
    epsilon: ArrayLike

    @beartype
    def __init__(
        self,
        wires: Sequence[int],
        epsilon: float,
        g: float,
        phi: float,
    ):
        super().__init__(wires=wires)
        self.epsilon = jnp.array(epsilon)
        self.g = jnp.array(g)
        self.phi = jnp.array(phi)
        return

    def __call__(self, dim: int):
        assert len(self.wires) == 2, "not correct wires"
        assert dim == 2, "not correct dim"
        rho = jnp.zeros(shape=(dim, dim, dim, dim), dtype=jnp.complex128)
        rho = rho.at[0, 0, 0, 0].set(1 - self.epsilon)
        rho = rho.at[0, 1, 0, 1].set(self.epsilon / 2)
        rho = rho.at[1, 0, 1, 0].set(self.epsilon / 2)
        rho = rho.at[0, 1, 1, 0].set(self.g * jnp.exp(1j * self.phi) * self.epsilon / 2)
        rho = rho.at[1, 0, 0, 1].set(
            self.g * jnp.exp(-1j * self.phi) * self.epsilon / 2
        )
        return rho


class S2(AbstractGate):
    r"""
    S2
    """

    r: ArrayLike
    phi: ArrayLike

    @beartype
    def __init__(self, wires: Sequence[int], r, phi):
        super().__init__(wires=wires)
        self.r = jnp.asarray(r)
        self.phi = jnp.asarray(phi)
        return

    def __call__(self, dim: int):
        s2_l = jnp.kron(create(dim), create(dim))
        s2_r = jnp.kron(destroy(dim), destroy(dim))
        u = jax.scipy.linalg.expm(
            1j * jnp.tanh(self.r) * (jnp.conjugate(self.phi) * s2_l - self.phi * s2_r)
        ).reshape(4 * (dim,))
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
        wires: tuple[int, int],
        r: float = jnp.pi / 4,
    ):
        super().__init__(wires=wires)
        self.r = jnp.array(r)
        return

    def __call__(self, dim: int):
        bs_l = jnp.kron(create(dim), destroy(dim))
        bs_r = jnp.kron(destroy(dim), create(dim))
        u = jax.scipy.linalg.expm(1j * self.r * (bs_l + bs_r)).reshape(4 * (dim,))
        return u
        # return einops.rearrange(u, "a b c d -> a c b d")


class LOPC(AbstractGate):
    r"""
    Linear optical passive element.
    """

    rs: ArrayLike

    @beartype
    def __init__(
        self,
        wires: tuple[int, ...],
        rs: Optional[ArrayLike] = None,
    ):
        super().__init__(wires=wires)
        if rs is None:
            rs = (
                jnp.ones(shape=[len(wires) * (len(wires) - 1) // 2], dtype=jnp.float64)
                * jnp.pi
                / 4
            )
        self.rs = jnp.array(rs)

    def __call__(self, dim: int):
        combs = list(itertools.combinations(range(len(self.wires)), 2))
        _h = sum(
            [
                functools.reduce(
                    jnp.kron,
                    [
                        {i: r * create(dim), j: destroy(dim)}.get(k, eye(dim))
                        for k in range(len(self.wires))
                    ],
                )
                for r, (i, j) in zip(self.rs, combs, strict=False)
            ]
            + [
                functools.reduce(
                    jnp.kron,
                    [
                        {j: r.conj() * create(dim), i: destroy(dim)}.get(k, eye(dim))
                        for k in range(len(self.wires))
                    ],
                )
                for r, (i, j) in zip(self.rs, combs, strict=False)
            ]
        )
        _s_matrix = (
            f"({' '.join([get_symbol(2 * k) for k in range(len(self.wires))])}) "
            f"({' '.join([get_symbol(2 * k + 1) for k in range(len(self.wires))])})"
        )
        # _s_tensor = f"{' '.join([get_symbol(2 * k) for k in range(len(self.wires))])} {' '.join([get_symbol(2 * k + 1) for k in range(len(self.wires))])}"
        _s_tensor = f"{' '.join([get_symbol(k) for k in range(2 * len(self.wires))])}"
        dims = {get_symbol(k): dim for k in range(2 * len(self.wires))}
        # print(_s_matrix)
        # print(_s_tensor)
        # u = jax.scipy.linalg.expm(
        #         1j * _h
        #     )
        # print(u.conj().T @ u)

        u = einops.rearrange(
            jax.scipy.linalg.expm(1j * _h), f"{_s_matrix} -> {_s_tensor}", **dims
        )
        return u


# %%

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
        wires: tuple[int] = (0,),
        phi: float | int = 0.0,
    ):
        super().__init__(wires=wires)
        self.phi = jnp.array(phi)
        return

    def __call__(self, dim: int):
        return jnp.diag(jnp.exp(1j * bases(dim) * self.phi))


fock_subtypes = {FockState, BeamSplitter, Phase}

# %%
if __name__ == "__main__":
    dim = 3
    wires = (0, 1, 2)
    coeff = jnp.pi / 2
    perms = list(
        itertools.permutations(
            [create(dim), destroy(dim)] + [eye(dim) for _ in range(len(wires) - 2)]
        )
    )
    terms = sum([functools.reduce(jnp.kron, perm) for perm in perms])
    u = jax.scipy.linalg.expm(1j * coeff * terms)
    # ket = functools.reduce(jnp.kron, (jnp.zeros(dim).at[1].set(1.0),) * 3)
    ket = functools.reduce(
        jnp.kron,
        [
            jnp.zeros(dim).at[0].set(1.0),
            jnp.zeros(dim).at[0].set(1.0),
            jnp.zeros(dim).at[1].set(1.0),
        ],
    )

    print(jnp.sum(jnp.abs(u @ ket) ** 2))
    u @ ket
    # %%
