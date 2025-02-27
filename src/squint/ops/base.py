# %%
import functools
import string
from string import ascii_lowercase, ascii_uppercase
from typing import Callable, Sequence, Union

import equinox as eqx
import jax.numpy as jnp
import scipy as sp
from beartype import beartype

characters = (
    string.ascii_lowercase
    + string.ascii_uppercase
    + "".join(chr(code) for code in range(0x03B1, 0x03C0))  # greek lowercase
    + "".join(chr(code) for code in range(0x0391, 0x03A0))  # greek uppercase
)


@functools.cache
def create(dim):
    return jnp.diag(jnp.sqrt(jnp.arange(1, dim)), k=-1)


@functools.cache
def destroy(dim):
    return jnp.diag(jnp.sqrt(jnp.arange(1, dim)), k=1)


@functools.cache
def eye(dim):
    return jnp.eye(dim)


@functools.cache
def bases(dim):
    return jnp.arange(dim)


@functools.cache
def dft(dim):
    return sp.linalg.dft(dim, scale="sqrtn")


class AbstractOp(eqx.Module):
    wires: tuple[int, ...]

    def __init__(
        self,
        wires=(0, 1),
    ):
        if not all([wire >= 0 for wire in wires]):
            raise TypeError("All wires must be nonnegative ints.")
        self.wires = wires
        return

    def unwrap(self):
        return (self,)


class AbstractState(AbstractOp):
    def __init__(
        self,
        wires=(0, 1),
    ):
        super().__init__(wires=wires)
        return

    def __call__(self, dim: int):
        return jnp.zeros(shape=(dim,) * len(self.wires))


class AbstractGate(AbstractOp):
    def __init__(
        self,
        wires=(0, 1),
    ):
        super().__init__(wires=wires)
        return

    def __call__(self, dim: int):
        left = ",".join(
            [ascii_lowercase[i] + ascii_uppercase[i] for i in range(len(self.wires))]
        )
        right = "".join(
            [ascii_lowercase[i] + ascii_uppercase[i] for i in range(len(self.wires))]
        )
        subscript = f"{left}->{right}"
        return jnp.einsum(
            subscript,
            *(
                [
                    jnp.eye(dim),
                ]
                * len(self.wires)
            ),
        )  # n-axis identity operator


class AbstractMeasurement(AbstractOp):
    def __init__(
        self,
        wires=(0, 1),
    ):
        super().__init__(wires=wires)
        return

    def __call__(self, dim: int):
        left = ",".join(
            [ascii_lowercase[i] + ascii_uppercase[i] for i in range(len(self.wires))]
        )
        right = "".join(
            [ascii_lowercase[i] + ascii_uppercase[i] for i in range(len(self.wires))]
        )
        subscript = f"{left}->{right}"
        return jnp.einsum(
            subscript,
            *(
                [
                    jnp.eye(dim),
                ]
                * len(self.wires)
            ),
        )  # n-axis identity operator


class SharedGate(AbstractGate):
    op: AbstractOp
    copies: Sequence[AbstractOp]
    where: Callable
    get: Callable

    @beartype
    def __init__(
        self,
        op: AbstractOp,
        wires: Union[Sequence[int], Sequence[Sequence[int]]],
        where: Callable = None,
        get: Callable = None,
    ):
        # todo: handle the wires coming from both main and the shared wires
        copies = [eqx.tree_at(lambda op: op.wires, op, (wire,)) for wire in wires]
        self.copies = copies
        self.op = op

        wires = op.wires + wires
        super().__init__(wires=wires)

        # use a default where/get sharing mechanism, such that all ArrayLike attributes are shared exactly
        attrs = [key for key, val in op.__dict__.items() if eqx.is_array_like(val)]

        if where is None:
            where = lambda pytree: sum(
                [[copy.__dict__[key] for copy in pytree.copies] for key in attrs], []
            )
        if get is None:
            get = lambda pytree: sum(
                [[pytree.op.__dict__[key] for _ in pytree.copies] for key in attrs], []
            )
        self.where = where
        self.get = get

        return

    # def __check_init__(self):
    #     return object.__setattr__(
    #         self,
    #         "__dict__",
    #         eqx.tree_at(self._where, self, replace_fn=lambda _: None).__dict__,
    #     )

    def unwrap(self):
        """Unwraps the shared ops for compilation and contractions."""
        _self = eqx.tree_at(
            self.where, self, self.get(self), is_leaf=lambda leaf: leaf is None
        )
        return [_self.op] + [op for op in _self.copies]
