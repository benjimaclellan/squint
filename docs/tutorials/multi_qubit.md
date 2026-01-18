# Entangled Sensors: Beating the Standard Quantum Limit

This tutorial explores how quantum entanglement can dramatically improve measurement precision beyond what's possible with classical or separable quantum states.

## The Power of Entanglement in Metrology

In the [single qubit tutorial](one_qubit.md), we saw that one qubit gives a Fisher Information of 1. What if we use $n$ independent qubits? Each contributes independently, giving a total Fisher Information of $n$. This is the **Standard Quantum Limit (SQL)**:

$$\mathcal{I}_\varphi^{\text{SQL}} = n$$

But quantum mechanics allows us to do better! By entangling our qubits, we can achieve the **Heisenberg Limit**:

$$\mathcal{I}_\varphi^{\text{HL}} = n^2$$

This quadratic improvement is one of the most celebrated results in quantum metrology.

---

## The GHZ State

The key to achieving the Heisenberg limit is the **Greenberger-Horne-Zeilinger (GHZ) state**, a maximally entangled state of $n$ qubits:

$$|\text{GHZ}\rangle_n = \frac{1}{\sqrt{2}}\left(|00\cdots0\rangle + |11\cdots1\rangle\right)$$

This state is a superposition of "all qubits in $|0\rangle$" and "all qubits in $|1\rangle$" — there's no classical analog for this kind of correlation.

### Why Does GHZ Give Quadratic Scaling?

When the phase $\varphi$ is encoded on each qubit via $R_z(\varphi)$, the GHZ state evolves to:

$$|\text{GHZ}(\varphi)\rangle = \frac{1}{\sqrt{2}}\left(|00\cdots0\rangle + e^{in\varphi}|11\cdots1\rangle\right)$$

The phase accumulates **coherently** across all $n$ qubits, giving a total phase of $n\varphi$ instead of just $\varphi$. This enhanced sensitivity directly translates to $n^2$ Fisher Information.

---

## Step 1: Creating Multiple Wires

For multi-qubit circuits, we create a list of wires:

```python
import jax.numpy as jnp
from squint.circuit import Circuit
from squint.ops.base import Wire, SharedGate
from squint.ops.dv import DiscreteVariableState, HGate, CXGate, RZGate
from squint.utils import partition_op

n_qubits = 4
wires = [Wire(dim=2, idx=i) for i in range(n_qubits)]
```

Each wire has a unique `idx` so Squint can track which operations act on which qubits.

---

## Step 2: Preparing the GHZ State

The GHZ state is prepared using a simple circuit:

1. Apply Hadamard to the first qubit: $|0\rangle \to \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$
2. Apply CNOT gates from the first qubit to all others

Let's build this step by step:

```python
circuit = Circuit(backend="pure")

# Initialize all qubits in |0⟩
for w in wires:
    circuit.add(DiscreteVariableState(wires=(w,), n=(0,)))
```

The circuit starts with the product state $|00\cdots0\rangle$.

### The Hadamard Gate

```python
circuit.add(HGate(wires=(wires[0],)))
```

After the Hadamard on qubit 0, we have:
$$\frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) \otimes |0\rangle \otimes \cdots \otimes |0\rangle$$

### The CNOT Cascade

```python
for i in range(1, n_qubits):
    circuit.add(CXGate(wires=(wires[0], wires[i])))
```

Each CNOT (controlled-X) gate entangles qubit 0 with qubit $i$:

- If qubit 0 is $|0\rangle$, qubit $i$ stays $|0\rangle$
- If qubit 0 is $|1\rangle$, qubit $i$ flips to $|1\rangle$

After all CNOTs, we have the GHZ state:
$$\frac{1}{\sqrt{2}}(|0000\rangle + |1111\rangle)$$

---

## Step 3: Encoding the Phase with SharedGate

Now we need to apply $R_z(\varphi)$ to **every** qubit. In metrology, all qubits sense the same physical parameter, so they should share the same phase value.

Squint provides `SharedGate` for exactly this purpose:

```python
circuit.add(
    SharedGate(
        op=RZGate(wires=(wires[0],), phi=0.0),
        wires=tuple(wires[1:])
    ),
    "phase"
)
```

This creates copies of the `RZGate` on all other wires, with the phase parameter **shared** across all copies. When we differentiate with respect to `phi`, Squint correctly accounts for the parameter appearing on all qubits.

---

## Step 4: Measurement Basis Rotation

To extract the phase information optimally, we apply Hadamard gates before measurement:

```python
for w in wires:
    circuit.add(HGate(wires=(w,)))
```

This converts the phase information encoded in the relative phase between $|00\cdots0\rangle$ and $|11\cdots1\rangle$ into population differences that can be measured in the computational basis.

---

## Step 5: Computing the Fisher Information

Now let's compile and compute the Fisher Information:

```python
params, static = partition_op(circuit, "phase")
sim = circuit.compile(static, params, optimize="greedy").jit()

qfi = sim.amplitudes.qfim(params)
cfi = sim.probabilities.cfim(params)

print(f"Number of qubits: {n_qubits}")
print(f"Standard Quantum Limit: {n_qubits}")
print(f"Heisenberg Limit: {n_qubits**2}")
print(f"Quantum Fisher Information: {qfi.squeeze():.1f}")
print(f"Classical Fisher Information: {cfi.squeeze():.1f}")
```

You should see that both QFI and CFI equal $n^2 = 16$ for 4 qubits — we've achieved the Heisenberg limit!

---

## Step 6: Verifying the Scaling

Let's verify the $n^2$ scaling by computing the Fisher Information for different numbers of qubits:

```python
import jax
import matplotlib.pyplot as plt

def compute_ghz_fisher_info(n):
    """Compute QFI for an n-qubit GHZ sensor."""
    wires = [Wire(dim=2, idx=i) for i in range(n)]
    circuit = Circuit(backend="pure")

    for w in wires:
        circuit.add(DiscreteVariableState(wires=(w,), n=(0,)))

    circuit.add(HGate(wires=(wires[0],)))
    for i in range(1, n):
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

# Compute for different system sizes
n_values = [1, 2, 3, 4, 5, 6]
qfi_values = [compute_ghz_fisher_info(n) for n in n_values]
```

```python
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(n_values, qfi_values, 'o-', label='GHZ (measured)', markersize=10)
ax.plot(n_values, [n**2 for n in n_values], '--', label='Heisenberg Limit ($n^2$)')
ax.plot(n_values, n_values, ':', label='Standard Quantum Limit ($n$)')

ax.set_xlabel('Number of qubits $n$')
ax.set_ylabel('Quantum Fisher Information')
ax.legend()
ax.set_xticks(n_values)

plt.tight_layout()
plt.show()
```

---

## The Physics Behind the Enhancement

Why does entanglement help? The key insight is that the GHZ state creates **quantum correlations** that don't exist in classical or separable states.

### Separable States (SQL)

With $n$ independent qubits, each in state $\frac{1}{\sqrt{2}}(|0\rangle + e^{i\varphi}|1\rangle)$, the total state is:

$$|\psi_{\text{sep}}\rangle = \bigotimes_{k=1}^n \frac{1}{\sqrt{2}}(|0\rangle + e^{i\varphi}|1\rangle)_k$$

The Fisher Information adds independently: $\mathcal{I} = n \times 1 = n$.

### Entangled States (Heisenberg Limit)

With the GHZ state, phase information is encoded in a **collective** degree of freedom:

$$|\text{GHZ}(\varphi)\rangle = \frac{1}{\sqrt{2}}(|00\cdots0\rangle + e^{in\varphi}|11\cdots1\rangle)$$

The effective phase is $n\varphi$, so the Fisher Information scales as $n^2$.

---

## Key Takeaways

1. **Entanglement enables quantum advantage** in metrology by allowing phase information to accumulate coherently

2. **GHZ states achieve the Heisenberg limit** with $\mathcal{I} = n^2$, a quadratic improvement over classical strategies

3. **SharedGate** in Squint handles parameter sharing across multiple qubits, essential for computing Fisher Information correctly

4. **The scaling is fragile** — as we'll see in the [noise tutorial](noise.md), decoherence can destroy the quantum advantage

---

## Next Steps

- [Noisy Sensors](noise.md): See how noise affects entanglement-enhanced sensing
- [Optimization](optimization.md): Learn to optimize probe states for maximum Fisher Information
