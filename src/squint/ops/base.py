# %%
import functools
import string
from string import ascii_lowercase, ascii_uppercase

import equinox as eqx
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import ArrayLike

# %%
characters = (
    string.ascii_lowercase
    + string.ascii_uppercase
    + "".join(chr(code) for code in range(0x03B1, 0x03C0))  # greek lowercase
    + "".join(chr(code) for code in range(0x0391, 0x03A0))  # greek uppercase
)


# %%
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


# %%
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


class Phase(AbstractGate):
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
