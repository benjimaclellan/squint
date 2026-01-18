# Single Qubit Phase Estimation

This tutorial introduces quantum metrology using `squint` by building a single-qubit phase sensor.

## The Protocol

In **phase estimation**, we estimate an unknown parameter $\varphi$ encoded as a phase rotation:

1. Prepare a probe state sensitive to $\varphi$
2. Encode $\varphi$ via a phase rotation
3. Measure and estimate

The precision is bounded by the **Fisher Information** $\mathcal{I}_\varphi$ through the Cramér-Rao bound: $\text{Var}(\hat{\varphi}) \geq 1/(N_{\text{meas}} \cdot \mathcal{I}_\varphi)$.

## Building the Circuit

We implement Ramsey interferometry: $|0\rangle \xrightarrow{H} \xrightarrow{R_z(\varphi)} \xrightarrow{H}$

```python
import jax.numpy as jnp
from squint.circuit import Circuit
from squint.ops.base import Wire
from squint.ops.dv import DiscreteVariableState, HGate, RZGate
from squint.utils import partition_op
```

In `squint`, quantum systems are built from `Wire` objects. For a qubit, use `dim=2`:

```python
wire = Wire(dim=2, idx=0)

circuit = Circuit(backend="pure")
circuit.add(DiscreteVariableState(wires=(wire,), n=(0,)))  # |0⟩
circuit.add(HGate(wires=(wire,)))                          # Hadamard
circuit.add(RZGate(wires=(wire,), phi=0.0), "phase")       # Phase rotation
circuit.add(HGate(wires=(wire,)))                          # Hadamard
```

The string key `"phase"` labels the operation for parameter partitioning.

## Compiling and Computing Fisher Information

Separate trainable parameters from the static structure, then compile:

```python
params, static = partition_op(circuit, "phase")
sim = circuit.compile(static, params, optimize="greedy").jit()
```

Compute the **Quantum Fisher Information** (ultimate precision limit) and **Classical Fisher Information** (precision with computational basis measurement):

```python
qfi = sim.amplitudes.qfim(params)
cfi = sim.probabilities.cfim(params)
print(f"QFI: {qfi.squeeze()}, CFI: {cfi.squeeze()}")
```

For Ramsey interferometry, both equal 1 — the measurement is optimal. This is the **Standard Quantum Limit (SQL)** for one qubit.

## Visualizing Phase Sensitivity

Sweep through phase values to see how probabilities and Fisher Information vary:

```python
import equinox as eqx
import jax
import matplotlib.pyplot as plt

phis = jnp.linspace(-jnp.pi, jnp.pi, 100)
params_sweep = eqx.tree_at(lambda p: p.ops["phase"].phi, params, phis)

probs = jax.vmap(sim.probabilities.forward)(params_sweep)
cfims = jax.vmap(sim.probabilities.cfim)(params_sweep)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
ax1.plot(phis, probs[:, 0], label=r'$p(0|\varphi)$')
ax1.plot(phis, probs[:, 1], label=r'$p(1|\varphi)$')
ax1.set_ylabel('Probability')
ax1.legend()

ax2.plot(phis, cfims.squeeze())
ax2.set_xlabel(r'Phase $\varphi$')
ax2.set_ylabel('Fisher Information')
```

The Fisher Information peaks where probabilities change most rapidly.

## Summary

- `Wire` objects define quantum subsystems (`dim=2` for qubits)
- `Circuit` holds operations added sequentially
- `partition_op` separates labeled parameters for differentiation
- QFI gives the ultimate precision; CFI depends on measurement choice
- Single qubit achieves $\mathcal{I} = 1$ (SQL)

Next: [Entangled Sensors](multi_qubit.md) shows how to beat the SQL with entanglement.
