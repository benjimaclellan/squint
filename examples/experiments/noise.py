# %%

import equinox as eqx

from squint.circuit import Circuit
from squint.ops.dv import DiscreteVariableState, XGate
from squint.ops.noise import BitFlipChannel
from squint.utils import print_nonzero_entries

# %%
dim = 2

circuit = Circuit()
n = 1

for i in range(n):
    circuit.add(DiscreteVariableState(wires=(i,)))
    circuit.add(XGate(wires=(i,)))
    # circuit.add(Phase(wires=(i,), phi=0.1))
    circuit.add(BitFlipChannel(wires=(i,), p=0.2))

# %%
path = circuit.path(dim=dim)
print(path)

# %%
params, static = eqx.partition(
    circuit,
    eqx.is_inexact_array,
)

sim = circuit.compile(params, static, dim=2)

# %%
print_nonzero_entries(sim.prob.forward(params))

# %%
# class TwoWireDepolarizing(AbstractChannel):
#     p: float

#     @beartype
#     def __init__(
#         self,
#         wires: tuple[int, int],
#         p: float
#     ):
#         super().__init__(wires=wires)
#         self.p = p  #paramax.non_trainable(p)
#         return

#     def __call__(self, dim: int):
#         assert dim == 2
#         return jnp.array(
#             [
#                 jnp.sqrt(1-self.p) * jnp.einsum('ac, bd-> abcd', basis_operators(dim=2)[3], basis_operators(dim=2)[3]),   # identity
#                 jnp.sqrt(self.p/3) * jnp.einsum('ac, bd-> abcd', basis_operators(dim=2)[0], basis_operators(dim=2)[0]),     # X
#                 jnp.sqrt(self.p/3) * jnp.einsum('ac, bd-> abcd', basis_operators(dim=2)[1], basis_operators(dim=2)[1]),    # X
#                 jnp.sqrt(self.p/3) * jnp.einsum('ac, bd-> abcd', basis_operators(dim=2)[2], basis_operators(dim=2)[2]),    # X
#             ]
#         )
