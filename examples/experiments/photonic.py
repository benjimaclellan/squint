# %%

from squint.circuit import Circuit
from squint.ops.dv import DiscreteState, X, Z

# %%
dim = 3
print(DiscreteState(wires=(0,), n=(1,)))
x = X(wires=(0,))
z = Z(wires=(0,))
print(x)
print(z)
print(x(dim=dim))
print(z(dim=dim))

circuit = Circuit()
circuit.add()

# %%
