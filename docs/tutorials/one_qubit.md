# Single Qubit Phase Estimation

This tutorial introduces quantum metrology using Squint by building a single-qubit phase estimation protocol from scratch.

## The Goal: Estimating an Unknown Phase

In quantum metrology, we use quantum systems to measure physical quantities with high precision. One of the most fundamental tasks is **phase estimation**: determining an unknown parameter $\varphi$ that appears as a phase rotation in our quantum system.

The general protocol consists of four stages:

1. **Prepare** a probe state sensitive to the parameter
2. **Encode** the parameter via a phase rotation
3. **Measure** the resulting quantum state
4. **Estimate** $\varphi$ from the measurement outcomes

The precision of our estimate is fundamentally limited by the **Fisher Information** — a quantity that tells us how much information our measurement contains about $\varphi$.

---

## Step 1: Setting Up the Environment

First, let's import the necessary modules:

```python
import jax.numpy as jnp
from squint.circuit import Circuit
from squint.ops.base import Wire
from squint.ops.dv import DiscreteVariableState, HGate, RZGate
from squint.utils import partition_op
```

In Squint, quantum systems are built from **Wires** — each wire represents a quantum subsystem with a specific dimension. For a qubit, we use dimension 2:

```python
wire = Wire(dim=2, idx=0)
```

The `idx` parameter is a unique identifier for this wire, which becomes important when we have multiple qubits.

---

## Step 2: Understanding the Circuit

We'll implement a Ramsey interferometry circuit, which is the standard approach for phase estimation with a single qubit:

$$|0\rangle \xrightarrow{H} \frac{|0\rangle + |1\rangle}{\sqrt{2}} \xrightarrow{R_z(\varphi)} \frac{|0\rangle + e^{i\varphi}|1\rangle}{\sqrt{2}} \xrightarrow{H} \cos(\varphi/2)|0\rangle + i\sin(\varphi/2)|1\rangle$$

Let's break down what happens at each step:

1. **Initial state**: We start in the ground state $|0\rangle$
2. **First Hadamard**: Creates an equal superposition — the probe state
3. **Phase rotation $R_z(\varphi)$**: Encodes the unknown parameter
4. **Second Hadamard**: Converts phase information into population differences

---

## Step 3: Building the Circuit in Squint

Now let's translate this into Squint code. We create a `Circuit` object and add operations sequentially:

```python
circuit = Circuit(backend="pure")
```

The `backend="pure"` tells Squint we're working with pure quantum states (state vectors). For noisy simulations, we'd use `backend="mixed"` for density matrices.

### Adding the Initial State

Every wire must start with a state. We initialize our qubit in $|0\rangle$:

```python
circuit.add(DiscreteVariableState(wires=(wire,), n=(0,)))
```

The `n=(0,)` specifies the computational basis state — here, the ground state.

### Adding the Gates

Now we add the Hadamard-Phase-Hadamard sequence:

```python
circuit.add(HGate(wires=(wire,)))                              # First Hadamard
circuit.add(RZGate(wires=(wire,), phi=0.0), "phase")           # Phase rotation
circuit.add(HGate(wires=(wire,)))                              # Second Hadamard
```

Notice the `"phase"` key on the RZ gate — this labels the operation so we can access its parameters later for optimization and Fisher information calculations.

---

## Step 4: Compiling the Circuit

Before we can simulate, we need to compile the circuit. This involves separating the **trainable parameters** (the phase $\varphi$) from the **static structure**:

```python
params, static = partition_op(circuit, "phase")
```

The `partition_op` function finds all operations with the key `"phase"` and extracts their parameters into a separate PyTree. This separation is essential for JAX's automatic differentiation.

Now we compile and JIT-compile for performance:

```python
sim = circuit.compile(static, params, optimize="greedy").jit()
```

The `optimize="greedy"` tells Squint to find an efficient tensor contraction order.

---

## Step 5: Computing the Fisher Information

The **Fisher Information** quantifies how sensitively our measurement probabilities change with $\varphi$. Higher Fisher Information means better estimation precision.

### Quantum Fisher Information (QFI)

The QFI represents the ultimate precision limit allowed by quantum mechanics:

$$\mathcal{I}_\varphi^{(Q)} = 4\left(\langle\partial_\varphi\psi|\partial_\varphi\psi\rangle - |\langle\psi|\partial_\varphi\psi\rangle|^2\right)$$

For our single qubit, the QFI equals 1 — this is the **Standard Quantum Limit**.

```python
qfi = sim.amplitudes.qfim(params)
print(f"Quantum Fisher Information: {qfi}")
```

### Classical Fisher Information (CFI)

The CFI depends on our choice of measurement basis:

$$\mathcal{I}_\varphi^{(C)} = \sum_i \frac{(\partial_\varphi p_i)^2}{p_i}$$

where $p_i = p(i|\varphi)$ are the measurement probabilities.

```python
cfi = sim.probabilities.cfim(params)
print(f"Classical Fisher Information: {cfi}")
```

For our Ramsey circuit with computational basis measurement, the CFI equals the QFI — our measurement is **optimal**!

---

## Step 6: Visualizing Phase Sensitivity

Let's see how the measurement probabilities and Fisher Information vary with $\varphi$:

```python
import equinox as eqx
import jax
import matplotlib.pyplot as plt

# Sweep through phase values from -π to π
phis = jnp.linspace(-jnp.pi, jnp.pi, 100)

# Update the phase parameter for each value
params_sweep = eqx.tree_at(lambda p: p.ops["phase"].phi, params, phis)

# Compute probabilities and Fisher Information for all phases
probs = jax.vmap(sim.probabilities.forward)(params_sweep)
cfims = jax.vmap(sim.probabilities.cfim)(params_sweep)
```

```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# Plot measurement probabilities
ax1.plot(phis, probs[:, 0], label=r'$p(0|\varphi)$')
ax1.plot(phis, probs[:, 1], label=r'$p(1|\varphi)$')
ax1.set_ylabel(r'Probability')
ax1.legend()

# Plot Fisher Information
ax2.plot(phis, cfims.squeeze())
ax2.set_xlabel(r'Phase $\varphi$')
ax2.set_ylabel(r'Fisher Information $\mathcal{I}_\varphi$')
ax2.axhline(y=1.0, color='gray', linestyle='--', label='SQL')

plt.tight_layout()
plt.show()
```

You'll notice that:

- The probabilities oscillate sinusoidally with $\varphi$
- The Fisher Information is **maximized** where the probabilities change most rapidly (steepest slope)
- The Fisher Information is **zero** at $\varphi = 0, \pm\pi$ where probabilities are flat

---

## Key Takeaways

1. **Wires** represent quantum subsystems; for qubits, use `dim=2`

2. **Circuits** are built by adding states and gates sequentially

3. **Parameter partitioning** separates trainable parameters for differentiation

4. **Fisher Information** quantifies estimation precision:
   - QFI: Ultimate quantum limit
   - CFI: Actual precision with a given measurement

5. The **Cramér-Rao bound** connects Fisher Information to estimation error:
   $$\text{Var}(\hat{\varphi}) \geq \frac{1}{N \cdot \mathcal{I}_\varphi}$$
   where $N$ is the number of measurements.

---

## Next Steps

- [Entangled Sensors](multi_qubit.md): Use entanglement to beat the Standard Quantum Limit
- [Noisy Sensors](noise.md): Model realistic noise and decoherence
- [Optimization](optimization.md): Find optimal probe states and measurements
