# %%
import copy
from collections import OrderedDict
from string import ascii_letters, ascii_lowercase, ascii_uppercase
from typing import OrderedDict, Sequence, Union

import equinox as eqx
import jax.numpy as jnp
import paramax
from beartype import beartype
from jaxtyping import ArrayLike


# %%
class AbstractOp(eqx.Module):
    wires: tuple[int, ...]

    def __init__(
        self,
        wires=(0, 1),
    ):
        self.wires = wires
        return


class AbstractState(AbstractOp):
    def __init__(
        self,
        wires=(0, 1),
    ):
        super().__init__(wires=wires)
        return

    def __call__(self, cut: int):
        return jnp.zeros(shape=(cut,) * len(self.wires))


class AbstractGate(AbstractOp):
    def __init__(
        self,
        wires=(0, 1),
    ):
        super().__init__(wires=wires)
        return

    def __call__(self, cut: int):
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
                    jnp.eye(cut),
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

    def __call__(self, cut: int):
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
                    jnp.eye(cut),
                ]
                * len(self.wires)
            ),
        )  # n-axis identity operator


# %%
class FockState(AbstractState):
    n: Union[
        Sequence[int], Sequence[tuple[complex, Sequence[int]]]
    ]  # todo: add superposition as n, using second typehint

    @beartype
    def __init__(
        self,
        wires=(0,),
        n: tuple[int] = (1,),
    ):
        super().__init__(wires=wires)
        self.n = paramax.non_trainable(n)
        return

    def __call__(self, cut: int):
        return jnp.zeros(shape=(cut,) * len(self.wires)).at[*list(self.n)].set(1.0)


class S2(AbstractGate):

    g: ArrayLike
    phi: ArrayLike

    @beartype
    def __init__(self, wires, g, phi):
        super().__init__(wires=wires)
        self.g = jnp.array(g)
        self.phi = jnp.array(phi)
        return


class BeamSplitter(AbstractGate):
    r: ArrayLike
    phi: ArrayLike

    @beartype
    def __init__(self, wires, r=jnp.array(jnp.pi / 4), phi=jnp.array(0.0)):
        super().__init__(wires=wires)
        self.r = jnp.array(r)
        self.phi = jnp.array(phi)
        return


class Phase(AbstractGate):
    phi: ArrayLike

    @beartype
    def __init__(
        self,
        wires=(0, 1),
        phi: float = 0.0,
    ):
        super().__init__(wires=wires)
        self.phi = jnp.array(phi)
        return


class Circuit(eqx.Module):
    ops: Sequence[AbstractOp]

    @beartype
    def __init__(
        self,
        # ops = OrderedDict, # | Sequence,
    ):
        self.ops = OrderedDict()

    @property
    def wires(self):
        return set(sum((op.wires for op in self.ops.values()), ()))

    def add(self, op: AbstractOp, key: str = None):  # todo:
        if key is None:
            key = len(self.ops)
        self.ops[key] = op

    @property
    def subscripts(self):
        chars = copy.copy(ascii_letters)
        _left_axes = []
        _right_axes = []
        _wire_chars = {wire: [] for wire in self.wires}
        for op in self.ops.values():
            _axis = []
            for wire in op.wires:
                if isinstance(op, AbstractState):
                    _left_axis = ""
                    _right_axes = chars[0]
                    chars = chars[1:]

                elif isinstance(op, AbstractGate):
                    _left_axis = _wire_chars[wire][-1]
                    _right_axes = chars[0]
                    chars = chars[1:]

                elif isinstance(op, AbstractMeasurement):
                    _left_axis = _wire_chars[wire][-1]
                    _right_axis = ""

                else:
                    raise TypeError

                _axis += [_left_axis, _right_axes]

                _wire_chars[wire].append(_right_axes)

            _left_axes.append("".join(_axis))

        _right_axes = [val[-1] for key, val in _wire_chars.items()]

        _left_expr = ",".join(_left_axes)
        _right_expr = "".join(_right_axes)
        subscripts = f"{_left_expr}->{_right_expr}"
        return subscripts
