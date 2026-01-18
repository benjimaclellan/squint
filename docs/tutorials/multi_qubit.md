# Entangled Sensors

This tutorial shows how entanglement improves measurement precision beyond the Standard Quantum Limit.

## SQL vs Heisenberg Limit

With $N$ independent qubits, Fisher Information adds linearly — the **Standard Quantum Limit (SQL)**:
$$\mathcal{I}_\varphi^{\text{SQL}} = N$$

With entanglement, we can achieve the **Heisenberg Limit**:
$$\mathcal{I}_\varphi^{\text{HL}} = N^2$$

The key is the **GHZ state**: $|\text{GHZ}\rangle = \frac{1}{\sqrt{2}}(|00\cdots0\rangle + |11\cdots1\rangle)$

When phase $\varphi$ is encoded on each qubit, the GHZ state evolves to:
$$|\text{GHZ}(\varphi)\rangle = \frac{1}{\sqrt{2}}(|00\cdots0\rangle + e^{iN\varphi}|11\cdots1\rangle)$$

The phase accumulates as $N\varphi$, giving $N^2$ Fisher Information.

## Building the GHZ Sensor

```python
import jax.numpy as jnp
from squint.circuit import Circuit
from squint.ops.base import Wire, SharedGate
from squint.ops.dv import DiscreteVariableState, HGate, CXGate, RZGate
from squint.utils import partition_op

N = 4
wires = [Wire(dim=2, idx=i) for i in range(N)]

circuit = Circuit(backend="pure")

# Initialize all qubits in |0⟩
for w in wires:
    circuit.add(DiscreteVariableState(wires=(w,), n=(0,)))

# GHZ preparation: H then CNOTs
circuit.add(HGate(wires=(wires[0],)))
for i in range(1, N):
    circuit.add(CXGate(wires=(wires[0], wires[i])))

# Phase encoding on all qubits (shared parameter)
circuit.add(
    SharedGate(op=RZGate(wires=(wires[0],), phi=0.0), wires=tuple(wires[1:])),
    "phase"
)

# Measurement basis
for w in wires:
    circuit.add(HGate(wires=(w,)))
```

`SharedGate` applies the same `RZGate` to all qubits with a **shared** phase parameter. This correctly accounts for the parameter appearing on all qubits when computing derivatives.

## Computing Fisher Information

```python
params, static = partition_op(circuit, "phase")
sim = circuit.compile(static, params, optimize="greedy").jit()

qfi = sim.amplitudes.qfim(params)
print(f"QFI: {qfi.squeeze():.0f}")
print(f"Heisenberg limit: {N**2}")
print(f"SQL: {N}")
```

For $N=4$, you should get QFI = 16 = $N^2$.

## Verifying the Scaling

```python
import matplotlib.pyplot as plt

def ghz_qfi(N):
    wires = [Wire(dim=2, idx=i) for i in range(N)]
    circuit = Circuit(backend="pure")

    for w in wires:
        circuit.add(DiscreteVariableState(wires=(w,), n=(0,)))

    circuit.add(HGate(wires=(wires[0],)))
    for i in range(1, N):
        circuit.add(CXGate(wires=(wires[0], wires[i])))

    circuit.add(
        SharedGate(op=RZGate(wires=(wires[0],), phi=0.0), wires=tuple(wires[1:])),
        "phase"
    )

    for w in wires:
        circuit.add(HGate(wires=(w,)))

    params, static = partition_op(circuit, "phase")
    sim = circuit.compile(static, params, optimize="greedy").jit()
    return sim.amplitudes.qfim(params).squeeze()

N_values = [1, 2, 3, 4, 5, 6]
qfi_values = [ghz_qfi(N) for N in N_values]

plt.plot(N_values, qfi_values, 'o-', label='GHZ', markersize=10)
plt.plot(N_values, [N**2 for N in N_values], '--', label='Heisenberg ($N^2$)')
plt.plot(N_values, N_values, ':', label='SQL ($N$)')
plt.xlabel('Number of qubits $N$')
plt.ylabel('Quantum Fisher Information')
plt.legend()
```

## Summary

- GHZ states achieve Heisenberg scaling: $\mathcal{I} = N^2$
- `SharedGate` handles parameter sharing across qubits
- The quadratic improvement comes from coherent phase accumulation
- This scaling is fragile to noise — see [Modeling Noise](noise.md)

Next: [Optimization](optimization.md) shows how to find optimal probe states variationally.
