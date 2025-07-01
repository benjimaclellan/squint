# Single qubit

This tutorial introduces the core concepts of quantum metrology using `squint`.
Let's build a simple quantum phase estimation protocol step-by-step for a single qubit. 

In quantum phase estimation, we want to estimate an unknown parameter $\varphi$ that appears as a phase rotation in our quantum system. The protocol consists of four stages:

1. **Prepare** a metrologically-useful probe state
2. **Interact** the probe with the unknown parameter $\varphi$
3. **Measure** the perturbed probe state
4. **Estimate** $\varphi$ from the measurement data

The initial state is,

$$|\psi\rangle =  |0\rangle $$

The quantum state evolves as,

$$|\psi(\varphi)\rangle = H \cdot R_z(\varphi) \cdot H \cdot |0\rangle$$

After the transformations, the state becomes,

$$|\psi(\varphi)\rangle = \cos(\varphi/2)|0\rangle + i\sin(\varphi/2)|1\rangle$$

Measuring in the $X$ basis (i.e., applying an H gate and measuring in the computational basis), the measurement probabilities are,

$$p(0|\varphi) = \cos^2(\varphi/2), \quad p(1|\varphi) = \sin^2(\varphi/2)$$

**Build a sensor in `squint`**

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from squint.circuit import Circuit
from squint.ops.dv import DiscreteVariableState, HGate, RZGate
from squint.utils import print_nonzero_entries, partition_op

# Create a simple one-qubit phase estimation circuit
# |0⟩ --- H --- Rz(φ) --- H --- |⟩
circuit = Circuit(backend="pure")
circuit.add(DiscreteVariableState(wires=(0,), n=(0,)))          # |0⟩ state
circuit.add(HGate(wires=(0,)))                                  # Hadamard gate
circuit.add(RZGate(wires=(0,), phi=0.0 * jnp.pi), "phase")      # Phase rotation
circuit.add(HGate(wires=(0,)))                                  # Second Hadamard

# Compile the circuit for simulation
dim = 2  # qubit dimension
params, static = partition_op(circuit, "phase")
sim = circuit.compile(static, dim, params, optimize="greedy").jit()
```


**Calculating the Fisher Information**

The Fisher Information quantifies how much information our measurements contain about $\varphi$:

```python
# Calculate quantum and classical Fisher Information
qfi = sim.amplitudes.qfim(params)       # Quantum Fisher Information
cfi = sim.probabilities.cfim(params)    # Classical Fisher Information

print(f"Quantum Fisher Information: {qfi}")
print(f"Classical Fisher Information: {cfi}")
```

The **Quantum Fisher Information (QFI)** for a pure state $|\psi(\varphi)\rangle$ is:

$$\mathcal{I}_\varphi^{(Q)} = 4(\langle\partial_\varphi\psi|\partial_\varphi\psi\rangle - |\langle\psi|\partial_\varphi\psi\rangle|^2)$$

The **Classical Fisher Information (CFI)** for measurement probabilities $p(s_i|\varphi)$ is:

$$\mathcal{I}_\varphi^{(C)} = \sum_i \frac{(\partial_\varphi p(s_i|\varphi))^2}{p(s_i|\varphi)}$$

**Visualizing the results**

```python
# Sweep through different phase values
phis = jnp.linspace(-jnp.pi, jnp.pi, 100)
params = eqx.tree_at(lambda pytree: pytree.ops["phase"].phi, params, phis)

probs = jax.vmap(sim.probabilities.forward)(params)
qfims = jax.vmap(sim.amplitudes.qfim)(params)
cfims = jax.vmap(sim.probabilities.cfim)(params)

# plot results
colors = sns.color_palette("Set2", n_colors=jnp.prod(jnp.array(probs.shape[1:])))
fig, axs = uplt.subplots(nrows=2, figsize=(6, 4), sharey=False)

for i, idx in enumerate(
    itertools.product(*[list(range(ell)) for ell in probs.shape[1:]])
):
    axs[0].plot(phis, probs[:, *idx], label=f"{idx}", color=colors[i])
axs[0].legend()
axs[0].set(xlabel=r"Phase, $\varphi$", ylabel=r"Probability, $p(\mathbf{x} | \varphi)$")

axs[1].plot(phis, qfims.squeeze(), color=colors[0], label=r"$\mathcal{I}_\varphi^Q$")
axs[1].plot(phis, cfims.squeeze(), color=colors[-1], label=r"$\mathcal{I}_\varphi^C$")
axs[1].set(
    xlabel=r"Phase, $\varphi$",
    ylabel=r"Fisher Information, $\mathcal{I}_\varphi$",
    ylim=[0, 1.05 * jnp.max(qfims)],
)
```

**Key takeaways**

- The Fisher Information tells us how precisely we can estimate $\varphi$
- Higher Fisher Information means better estimation precision
- The Cramér-Rao bound gives us: $\Delta^2\bar{\varphi} \geq 1/\mathcal{I}_\varphi$
