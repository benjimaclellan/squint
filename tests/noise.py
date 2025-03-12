#%%
import itertools
import functools 

import einops
import equinox as eqx
import jax
import optax
import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from rich.pretty import pprint
import polars as pl
from beartype import beartype
from opt_einsum.parser import get_symbol

from squint.ops.base import AbstractChannel, basis_operators, AbstractGate, AbstractState
from squint.ops.dv import DiscreteState, XGate, ZGate
from squint.circuit import Circuit
from squint.utils import print_nonzero_entries

#%%
rho = jnp.ones(shape=(2,))

channel = jnp.array([
    jnp.ones(shape=(2, 2)),
    jnp.ones(shape=(2, 2)),
    jnp.ones(shape=(2, 2)),
])

rho_ch = jnp.einsum('a, x a b -> xb', rho, channel)
print(rho_ch)
einops.reduce(rho_ch, 'x b -> b', 'sum')

#%%
rho = jnp.ones(shape=(2, 2))
channel = jnp.array([
    jnp.ones(shape=(2, 2, 2, 2)),
    jnp.ones(shape=(2, 2, 2, 2)),
    jnp.ones(shape=(2, 2, 2, 2)),
])
rho_ch = jnp.einsum('a b, x a b c d -> x c d', rho, channel)
print(rho_ch)
einops.reduce(rho_ch, 'x c d -> c d', 'sum')



#%%
class BitFlipChannel(AbstractChannel):
    p: float 
    
    @beartype
    def __init__(
        self,
        wires: tuple[int],
        p: float
    ):
        super().__init__(wires=wires)
        self.p = p  #paramax.non_trainable(p)
        return

    def __call__(self, dim: int):
        assert dim == 2
        return jnp.array(
            [
                jnp.sqrt(1-self.p) * basis_operators(dim=2)[3],   # identity
                jnp.sqrt(self.p) * basis_operators(dim=2)[2],     # X
            ]
        )

channel = BitFlipChannel(wires=(0,), p=1.0)
channel(dim=2)

state = DiscreteState(wires=(0,))
rho = jnp.einsum('a, b -> a b', state(dim=2), state(dim=2).conj())

einops.reduce(
    jnp.einsum('xab, bc, xcd  -> xad', channel(dim=2), rho, channel(dim=2).conj()), 
    'x a d -> a d', 
    'sum'
)

#%%
dim = 2

circuit = Circuit()
# circuit.add(DiscreteState(wires=(0, 1)))
n = 2

for i in range(n):
    
    circuit.add(DiscreteState(wires=(i,)))
    circuit.add(XGate(wires=(i,)))
    circuit.add(BitFlipChannel(wires=(i,), p=0.2))

# circuit.add(DiscreteState(wires=(1,)))
# circuit.add(XGate(wires=(0,)))
# circuit.add(XGate(wires=(1,)))
# circuit.add(BitFlipChannel(wires=(0,), p=0.2))
# circuit.add(BitFlipChannel(wires=(1,), p=0.2))
# circuit.add(ZGate(wires=(0,)))

# state = DiscreteState(wires=(0,))
# rho = jnp.einsum('a,b->ab', state(dim=dim), state(dim=dim).conj())

# u = XGate(wires=(0,))
# channel = BitFlipChannel(wires=(0,), p=0.5)

# def unwrap_dm(circuit, dim):
#     _tensors = []
#     for op in circuit.unwrap():
#         print(op)
#         if isinstance(op, AbstractState):
#             _subscripts_l = (
#                 ''.join([get_symbol(2 * i) for i in range(len(op.wires))])
#                 + ','
#                 + ''.join([get_symbol(2 * i + 1) for i in range(len(op.wires))])
#             )
#             _subscripts_r = (
#                 ''.join([get_symbol(2 * i) for i in range(len(op.wires))])
#                 + ''.join([get_symbol(2 * i + 1) for i in range(len(op.wires))])
#             )
#             # print(f"{_subscripts_l} -> {_subscripts_r}")
#             _state = op(dim=dim)
#             _tensor = jnp.einsum(f"{_subscripts_l} -> {_subscripts_r}", _state, _state.conj())
#         else:
#             _tensor = op(dim=dim)
#         _tensors.append(_tensor)
#     return _tensors

# # unwrap tensors,
# # put tensors in a canonical ordering 
# # set up left mirror leg characters, right mirror leg characters

# _tensors = [channel(dim).conj().astype('complex64'), u(dim).conj().astype('complex64'), rho.astype('complex64'), u(dim).astype('complex64'), channel(dim).astype('complex64')]
# # _tensors = unwrap_dm(circuit, dim)
# subscripts = 'xce, ac, ab, bd, xdf -> xef'
# einops.reduce(jnp.einsum(subscripts, *_tensors), 'x a b -> a b', 'sum')

#%%
START_RIGHT = 0
# START_LEFT = 10000
START_CHANNEL = 20000


def get_symbol_right(i):
    # assert i + START_RIGHT < START_LEFT, "Collision of leg symbols"
    return get_symbol(2*i)
    # return get_symbol(i + START_RIGHT)

def get_symbol_left(i):
    # assert i + START_LEFT < START_CHANNEL, "Collision of leg symbols"
    return get_symbol(2*i + 1)
    # return get_symbol(i + START_LEFT)

def get_symbol_channel(i):
    # assert i + START_CHANNEL < START_LEFT, "Collision of leg symbols"
    return get_symbol(2 * i + START_CHANNEL)

def unwrap_dm(circuit, dim):
    _tensors_right = [op(dim) for op in circuit.unwrap()]
    _tensors_left = [op(dim).conj() for op in circuit.unwrap()]
        

# get_symbol_left(10)


def subscripts(circuit, get_symbol, get_symbol_channel):
    _iterator = itertools.count(0)
    _iterator_channel = itertools.count(0)

    _in_axes = []
    _out_axes = []
    _wire_chars = {wire: [] for wire in circuit.wires}
    for op in circuit.unwrap():
        _axis = []
        for wire in op.wires:
            if isinstance(op, AbstractState):
                _in_axis = ""
                _out_axes = get_symbol(next(_iterator))

            elif isinstance(op, AbstractGate):
                _in_axis = _wire_chars[wire][-1]
                _out_axes = get_symbol(next(_iterator))

            elif isinstance(op, AbstractChannel):
                _in_axis = _wire_chars[wire][-1]
                _out_axes = get_symbol(next(_iterator))

            elif isinstance(op, AbstractMeasurement):
                _in_axis = _wire_chars[wire][-1]
                _right_axis = ""
            
            else:
                raise TypeError

            _axis += [_in_axis, _out_axes]

            _wire_chars[wire].append(_out_axes)

        # add extra axis for channel
        if isinstance(op, AbstractChannel):
            _axis.insert(0, get_symbol_channel(next(_iterator_channel)))
            
        _in_axes.append("".join(_axis))

    _out_axes = [val[-1] for key, val in _wire_chars.items()]

    _in_expr = ",".join(_in_axes)
    _out_expr = "".join(_out_axes)
    _subscripts = f"{_in_expr}->{_out_expr}"
    return _in_expr, _out_expr


_in_expr_ket, _out_expr_ket = subscripts(circuit, get_symbol_right, get_symbol_channel)
_in_expr_bra, _out_expr_bra = subscripts(circuit, get_symbol_left, get_symbol_channel)

_tensors_ket = [op(dim=dim) for op in circuit.unwrap()]
_tensors_bra = [op(dim=dim).conj() for op in circuit.unwrap()]


print(_in_expr_ket, _out_expr_ket)
print(_in_expr_bra, _out_expr_bra)
_subscripts = f"{_in_expr_ket},{_in_expr_bra}->{_out_expr_ket}{_out_expr_bra}"
print(_subscripts)

state = jnp.einsum(_subscripts, *(_tensors_ket + _tensors_bra), optimize='greedy')
print_nonzero_entries(state)

#%%
_tensors = [op(dim=dim) for op in circuit.unwrap()]
for op in circuit.unwrap():
    print(op)
    
#%%

a = jnp.ones(shape=(5,))
jnp.einsum('a -> ', a)


# %%
class TwoWireDepolarizing(AbstractChannel):
    p: float 
    
    @beartype
    def __init__(
        self,
        wires: tuple[int, int],
        p: float
    ):
        super().__init__(wires=wires)
        self.p = p  #paramax.non_trainable(p)
        return

    def __call__(self, dim: int):
        assert dim == 2
        return jnp.array(
            [
                jnp.sqrt(1-self.p) * jnp.einsum('ac, bd-> abcd', basis_operators(dim=2)[3], basis_operators(dim=2)[3]),   # identity
                jnp.sqrt(self.p/3) * jnp.einsum('ac, bd-> abcd', basis_operators(dim=2)[0], basis_operators(dim=2)[0]),     # X
                jnp.sqrt(self.p/3) * jnp.einsum('ac, bd-> abcd', basis_operators(dim=2)[1], basis_operators(dim=2)[1]),    # X
                jnp.sqrt(self.p/3) * jnp.einsum('ac, bd-> abcd', basis_operators(dim=2)[2], basis_operators(dim=2)[2]),    # X
            ]
        )


state = DiscreteState(wires=(0, 1))
rho = jnp.einsum('cd, ef -> cdef', state(dim=2), state(dim=2).conj())

channel = TwoWireDepolarizing(wires=(0, 1), p=1.0)
channel(dim=2).shape

rho_prime = einops.reduce(
    jnp.einsum('xabcd, cdef, xefgh  -> xabgh', channel(dim=2), rho, channel(dim=2).conj()), 
    'x a b g h -> a b g h',
    'sum'
)
print(rho_prime)
# %%

import jax.random as jr

key = jr.PRNGKey(1)
mat_a = jr.normal(key, shape=(4, 4))
mat_b = jr.normal(jr.split(key)[0], shape=(4, 4))

mat_c = mat_a @ mat_b

ten_a = mat_a.reshape((2, 2, 2, 2))
ten_b = mat_b.reshape((2, 2, 2, 2))
ten_c = jnp.einsum('abcd, cdef -> abef', ten_a, ten_b)

ten_c.reshape((4, 4))

print(ten_c)
print("\n")
print(mat_c.reshape((2, 2, 2, 2)))

#%%
key = jr.PRNGKey(1)
mat_a = jr.normal(key, shape=(3, 4, 4))
mat_b = jr.normal(jr.split(key)[0], shape=(3, 4, 4))

mat_c = jnp.stack([mat_a[i] @ mat_b[i] for i in range(3)])

ten_a = mat_a.reshape((3, 2, 2, 2, 2))
ten_b = mat_b.reshape((3, 2, 2, 2, 2))
ten_c = jnp.einsum('xabcd, xcdef -> xabef', ten_a, ten_b)

ten_c.reshape((3, 4, 4))

print(ten_c)
print("\n")
print(mat_c.reshape((3, 2, 2, 2, 2)))

jnp.all(ten_c == mat_c.reshape((3, 2, 2, 2, 2)))