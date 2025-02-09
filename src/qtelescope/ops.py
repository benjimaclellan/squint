# %%
import copy
import functools
import string
from collections import OrderedDict
from dataclasses import dataclass
from string import ascii_lowercase, ascii_uppercase
from typing import Any, Callable, Sequence

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import paramax
from beartype import beartype
from beartype.door import is_bearable
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
def create(cut):
    return jnp.diag(jnp.sqrt(jnp.arange(1, cut)), k=-1)


@functools.cache
def destroy(cut):
    return jnp.diag(jnp.sqrt(jnp.arange(1, cut)), k=1)


@functools.cache
def bases(cut):
    return jnp.arange(cut)


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
    n: Sequence[
        tuple[complex, Sequence[int]]
    ]  # todo: add superposition as n, using second typehint

    @beartype
    def __init__(
        self,
        wires: Sequence[int],
        n: Sequence[int] | Sequence[tuple[complex | float, Sequence[int]]] = None,
    ):
        super().__init__(wires=wires)
        if n is None:
            n = [(1.0, (1,) * len(wires))]  # initialize to |1, 1, ...> state
        if is_bearable(n, Sequence[int]):
            n = [(1.0, n)]
        self.n = paramax.non_trainable(n)
        return

    def __call__(self, cut: int):
        return sum(
            [
                jnp.zeros(shape=(cut,) * len(self.wires)).at[*term[1]].set(term[0])
                for term in self.n
            ]
        )


# op = FockState(wires=(0, 1), n=[(1.0, (1,1))])#(cut=4)
# print(op(cut=4))


# %%
class S2(AbstractGate):
    g: ArrayLike
    phi: ArrayLike

    @beartype
    def __init__(self, wires: Sequence[int], g, phi):
        super().__init__(wires=wires)
        self.g = jnp.array(g)
        self.phi = jnp.array(phi)
        return


class BeamSplitter(AbstractGate):
    r: ArrayLike
    phi: ArrayLike

    @beartype
    def __init__(
        self,
        wires: tuple[int, int],
        r: float = jnp.pi / 4,
        phi: float = 0.0,
    ):
        super().__init__(wires=wires)
        self.r = jnp.array(r)
        self.phi = jnp.array(phi)
        return

    def __call__(self, cut: int):
        create(cut)
        # bs_l = jnp.einsum("ab,cd->abcd", create(cut), destroy(cut))
        # bs_r = jnp.einsum("ab,cd->abcd", destroy(cut), create(cut))
        bs_l = jnp.kron(create(cut), destroy(cut))
        bs_r = jnp.kron(destroy(cut), create(cut))

        u = jax.scipy.linalg.expm(1j * self.r * (bs_l + bs_r)).reshape(4 * (cut,))
        # self.r * (jnp.exp(1j * self.phi * bs_l - jnp.exp(-1j * self.phi) * bs_r)
        # return u
        return einops.rearrange(u, "a b c d -> a c b d")


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

    def __call__(self, cut: int):
        return jnp.diag(jnp.exp(1j * bases(cut) * self.phi))


class Circuit(eqx.Module):
    ops: Sequence[AbstractOp]

    @beartype
    def __init__(
        self,
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
        chars = copy.copy(characters)
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

    def path(self, cut: int, optimize: str = "greedy", verbose: bool = False):
        path, info = jnp.einsum_path(
            self.subscripts,
            *[op(cut=cut) for op in self.ops.values()],
            optimize=optimize,
        )
        if verbose:
            print(info)

        return path, info

    def compile(
        self, params, static, cut: int, optimize: str = "greedy", verbose: bool = False
    ):
        path, info = self.path(cut=cut, optimize=optimize, verbose=verbose)

        def _tensor_func(circuit, subscripts: str, optimize: tuple):
            return jnp.einsum(
                subscripts,
                *[op(cut=cut) for op in circuit.ops.values()],
                optimize=optimize,
            )

        _tensor = functools.partial(
            _tensor_func, subscripts=self.subscripts, optimize=path
        )

        def _forward_func(params, static):
            circuit = paramax.unwrap(eqx.combine(params, static))
            return _tensor(circuit)

        _forward = functools.partial(_forward_func, static=static)

        def _probability(params):
            state = _forward(params)
            return jnp.abs(state) ** 2

        _grad = jax.jacrev(_probability)
        _hess = jax.jacfwd(_grad)

        return Simulator(
            forward=_forward,
            probability=_probability,
            grad=_grad,
            hess=_hess,
            path=path,
            info=info,
        )


@dataclass
class Simulator:
    forward: Callable
    probability: Callable
    grad: Callable
    hess: Callable
    path: Any
    info: str = None

    def jit(self):
        return Simulator(
            forward=jax.jit(self.forward),
            probability=jax.jit(self.probability),
            grad=jax.jit(self.grad),
            hess=jax.jit(self.hess),
            path=self.path,
            info=self.info,
        )
