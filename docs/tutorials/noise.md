# Modeling Noise

Real quantum devices are imperfect. This tutorial shows how to model noise and decoherence in `squint`, and explores how noise affects the quantum advantage from entanglement.

## From Pure States to Density Matrices

To model noise, we need **density matrices** rather than state vectors. In `squint`, the backend is automatically selected based on the operations in your circuit. When you add noise channels or mixed states, the mixed backend is used automatically:

```python
from squint.circuit import Circuit
circuit = Circuit()  # Backend auto-selected based on operations
```

Noise is modeled using **quantum channels** with Kraus operators $\{K_i\}$:

$$\rho \to \mathcal{E}(\rho) = \sum_i K_i \rho K_i^\dagger$$

`squint` provides several built-in channels:

| Channel | Kraus Operators |
|---------|-----------------|
| `BitFlipChannel` | $K_0 = \sqrt{1-p}\,I$, $K_1 = \sqrt{p}\,X$ |
| `PhaseFlipChannel` | $K_0 = \sqrt{1-p}\,I$, $K_1 = \sqrt{p}\,Z$ |
| `DepolarizingChannel` | $K_0 = \sqrt{1-3p/4}\,I$, $K_{1,2,3} = \sqrt{p/4}\,\{X,Y,Z\}$ |

## Building a Noisy GHZ Sensor

```python
import jax.numpy as jnp
from squint.circuit import Circuit
from squint.simulator.tn import Simulator
from squint.ops.base import Wire, SharedGate
from squint.ops.dv import DiscreteVariableState, HGate, CXGate, RZGate
from squint.ops.noise import DepolarizingChannel
from squint.utils import partition_op

N = 4
noise_p = 0.05
wires = [Wire(dim=2, idx=i) for i in range(N)]

circuit = Circuit()

# Initialize qubits
for w in wires:
    circuit.add(DiscreteVariableState(wires=(w,), n=(0,)))

# GHZ preparation with noise after each gate
circuit.add(HGate(wires=(wires[0],)))
circuit.add(DepolarizingChannel(wires=(wires[0],), p=noise_p))

for i in range(1, N):
    circuit.add(CXGate(wires=(wires[0], wires[i])))
    circuit.add(DepolarizingChannel(wires=(wires[0],), p=noise_p))
    circuit.add(DepolarizingChannel(wires=(wires[i],), p=noise_p))

# Phase encoding
circuit.add(
    SharedGate(op=RZGate(wires=(wires[0],), phi=0.0), wires=tuple(wires[1:])),
    "phase"
)

# Measurement
for w in wires:
    circuit.add(HGate(wires=(w,)))
```

## Computing Fisher Information

```python
params, static = partition_op(circuit, "phase")
sim = Simulator.compile(static, params, optimize="greedy").jit()

cfi = sim.probabilities.cfim(params)
print(f"Noisy CFI: {cfi.squeeze():.2f}")
print(f"Heisenberg limit: {N**2}")
```

## Noise Destroys Heisenberg Scaling

A key result: **constant noise per qubit destroys Heisenberg scaling**. Let's verify:

```python
import matplotlib.pyplot as plt

def noisy_cfi(N, noise_p):
    wires = [Wire(dim=2, idx=i) for i in range(N)]
    circuit = Circuit()

    for w in wires:
        circuit.add(DiscreteVariableState(wires=(w,), n=(0,)))

    circuit.add(HGate(wires=(wires[0],)))
    circuit.add(DepolarizingChannel(wires=(wires[0],), p=noise_p))

    for i in range(1, N):
        circuit.add(CXGate(wires=(wires[0], wires[i])))
        circuit.add(DepolarizingChannel(wires=(wires[0],), p=noise_p))
        circuit.add(DepolarizingChannel(wires=(wires[i],), p=noise_p))

    circuit.add(
        SharedGate(op=RZGate(wires=(wires[0],), phi=0.0), wires=tuple(wires[1:])),
        "phase"
    )

    for w in wires:
        circuit.add(HGate(wires=(w,)))

    params, static = partition_op(circuit, "phase")
    sim = Simulator.compile(static, params, optimize="greedy").jit()
    return sim.probabilities.cfim(params).squeeze()

# Vary system size
N_values = [2, 3, 4, 5, 6]
cfi_noisy = [noisy_cfi(N, 0.02) for N in N_values]

plt.plot(N_values, cfi_noisy, 'o-', label='Noisy GHZ')
plt.plot(N_values, [N**2 for N in N_values], '--', label='Heisenberg')
plt.plot(N_values, N_values, ':', label='SQL')
plt.xlabel('Number of qubits $N$')
plt.ylabel('Fisher Information')
plt.legend()
```

For small $N$, GHZ beats the SQL. For large $N$, noise accumulates and GHZ performs *worse* than separable states. There's an optimal system size balancing quantum enhancement against noise.

## Other Noise Channels

```python
from squint.ops.noise import BitFlipChannel, PhaseFlipChannel, ErasureChannel

# Bit flip (random X errors)
circuit.add(BitFlipChannel(wires=(wire,), p=0.1))

# Phase flip / dephasing (random Z errors)
circuit.add(PhaseFlipChannel(wires=(wire,), p=0.1))

# Erasure (traces out the qubit)
circuit.add(ErasureChannel(wires=(wire,)))
```

## Summary

- Add noise channels to your circuit and the mixed backend is automatically used
- Noise channels transform $\rho \to \sum_i K_i \rho K_i^\dagger$
- GHZ states are fragile: Heisenberg scaling is lost with constant per-qubit noise
- The [optimization tutorial](optimization.md) shows how to find noise-robust states
