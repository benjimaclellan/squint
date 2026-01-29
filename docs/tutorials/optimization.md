# Optimizing Quantum Sensors

This tutorial shows how to find optimal probe states using gradient-based optimization. We maximize Fisher Information with respect to trainable gate angles using JAX autodiff and Optax.

## Variational Quantum Metrology

In previous tutorials we used fixed probe states like GHZ. **Variational quantum metrology** instead parameterizes the circuit and optimizes to maximize Fisher Information:
$$\min_{\theta} \left[ -\mathcal{I}_\varphi(\theta) \right]$$

This discovers optimal probes for specific noise models, hardware constraints, and measurement strategies.

## Building a Variational Sensor

The **hardware-efficient ansatz** alternates single-qubit rotations with entangling gates:

```python
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from squint.circuit import Circuit
from squint.simulator.tn import Simulator
from squint.ops.base import Wire, SharedGate
from squint.ops.dv import DiscreteVariableState, RXGate, RYGate, RZGate, CXGate
from squint.utils import partition_op

N = 4  # qubits
n_layers = 2
wires = [Wire(dim=2, idx=i) for i in range(N)]
circuit = Circuit()

# Initialize |0⟩^N
for w in wires:
    circuit.add(DiscreteVariableState(wires=(w,), n=(0,)))

# Variational layers: rotations + entanglement
for layer in range(n_layers):
    for i, w in enumerate(wires):
        circuit.add(RXGate(wires=(w,), phi=0.1), f"rx_{layer}_{i}")
        circuit.add(RYGate(wires=(w,), phi=0.1), f"ry_{layer}_{i}")
    for i in range(N - 1):
        circuit.add(CXGate(wires=(wires[i], wires[i + 1])))

# Phase encoding (estimation target)
circuit.add(
    SharedGate(op=RZGate(wires=(wires[0],), phi=0.0), wires=tuple(wires[1:])),
    "phase"
)

# Measurement basis rotations
for i, w in enumerate(wires):
    circuit.add(RXGate(wires=(w,), phi=0.1), f"meas_rx_{i}")
    circuit.add(RYGate(wires=(w,), phi=0.1), f"meas_ry_{i}")
```

String keys like `"rx_0_1"` label trainable gates for partitioning.

## Parameter Partitioning

We separate estimation parameters (the phase we want to estimate) from optimization parameters (gate angles we train):

```python
params, static = partition_op(circuit, "phase")
opt_params, opt_static = eqx.partition(static, eqx.is_inexact_array)
sim = Simulator.compile(opt_static, params, opt_params, optimize="greedy").jit()
```

Now `params` holds the phase, `opt_params` holds trainable angles, and `opt_static` holds non-trainable structure.

## Optimization Loop

Define the loss as negative CFI and optimize with Adam:

```python
def loss_fn(params, opt_params):
    return -sim.probabilities.cfim(params, opt_params).squeeze()

optimizer = optax.adam(0.05)
opt_state = optimizer.init(opt_params)

@jax.jit
def step(params, opt_params, opt_state):
    loss, grad = jax.value_and_grad(loss_fn, argnums=1)(params, opt_params)
    updates, opt_state = optimizer.update(grad, opt_state, opt_params)
    opt_params = optax.apply_updates(opt_params, updates)
    return opt_params, opt_state, loss

for i in range(200):
    opt_params, opt_state, loss = step(params, opt_params, opt_state)
    if i % 50 == 0:
        print(f"Step {i}: CFI = {-loss:.2f}")

print(f"Final: CFI = {-loss:.2f}, Heisenberg = {N**2}, SQL = {N}")
```

With sufficient depth, the CFI approaches the Heisenberg limit $N^2$.

## Visualization

```python
import matplotlib.pyplot as plt

losses = []
opt_params, opt_static = eqx.partition(static, eqx.is_inexact_array)
opt_state = optimizer.init(opt_params)

for i in range(200):
    opt_params, opt_state, loss = step(params, opt_params, opt_state)
    losses.append(-loss)

plt.plot(losses, label='Optimized CFI')
plt.axhline(y=N**2, color='green', linestyle='--', label=f'Heisenberg ({N**2})')
plt.axhline(y=N, color='orange', linestyle='--', label=f'SQL ({N})')
plt.xlabel('Step')
plt.ylabel('CFI')
plt.legend()
```

## Optimization with Noise

The same approach works with noisy circuits (the mixed backend is automatically selected when noise channels are present):

```python
from squint.ops.noise import DepolarizingChannel

noise_p = 0.02
circuit = Circuit()

for w in wires:
    circuit.add(DiscreteVariableState(wires=(w,), n=(0,)))

for layer in range(n_layers):
    for i, w in enumerate(wires):
        circuit.add(RXGate(wires=(w,), phi=0.0), f"rx_{layer}_{i}")
        circuit.add(DepolarizingChannel(wires=(w,), p=noise_p))
    for i in range(N - 1):
        circuit.add(CXGate(wires=(wires[i], wires[i + 1])))

circuit.add(SharedGate(op=RZGate(wires=(wires[0],), phi=0.0), wires=tuple(wires[1:])), "phase")
```

Optimization can discover **noise-resilient** probes that outperform GHZ under realistic conditions.

## Tips

- **Learning rate**: Start with 0.01-0.1
- **Random initialization**: Helps escape local minima
  ```python
  import jax.random as jr
  opt_params = jax.tree.map(
      lambda x: jr.uniform(jr.PRNGKey(42), x.shape, minval=-0.1, maxval=0.1),
      opt_params
  )
  ```
- **Gradient clipping**: `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))`
- **Multiple runs**: Loss landscape may have local minima

## Summary

- Variational optimization finds optimal probes by maximizing Fisher Information
- `partition_op` separates estimation from trainable parameters
- JAX autodiff computes gradients through the quantum simulation
- The framework extends naturally to noisy circuits
