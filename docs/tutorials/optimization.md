
# Optimization

This guide shows you how to optimization a parameterized `squint` circuit.

```python
import optax
from squint.circuit import Circuit
from squint.base import SharedGate
from squint.ops.dv import DiscreteVariableState, RXGate, RYGate, RZGate, CXGate
from squint.utils import partition_op

def variational_sensor(n_qubits, n_layers):
    """Create a hardware-efficient ansatz variational quantum sensor."""
    circuit = Circuit(backend="pure")
    
    # Initialize qubits
    for i in range(n_qubits):
        circuit.add(DiscreteVariableState(wires=(i,), n=(0,)))
    
    # Variational layers
    for layer in range(n_layers):
        # Single-qubit rotations
        for i in range(n_qubits):
            circuit.add(RXGate(wires=(i,), phi=0.0), f"rx_{layer}_{i}")
            circuit.add(RYGate(wires=(i,), phi=0.0), f"ry_{layer}_{i}")
        
        # Entangling layer
        for i in range(n_qubits - 1):
            circuit.add(CXGate(wires=(i, i+1)))
    
    # Phase sensing layer
    circuit.add(
        SharedGate(op=RZGate(wires=(0,), phi=0.0 * jnp.pi), wires=tuple(range(1, n_qubits))),
        "phase",
    )

    # Measurement layer
    for i in range(n_qubits):
        circuit.add(RXGate(wires=(i,), phi=0.0), f"meas_x_{i}")
        circuit.add(RYGate(wires=(i,), phi=0.0), f"meas_y_{i}")
    
    return circuit
```

**Optimization loop**

```python
n_qubits = 4
n_layers = 3
n_steps = 100

circuit = variational_sensor(n_qubits, n_layers)

params, static = eqx.partition(circuit, eqx.is_inexact_array)
params_est, params_opt = partition_op(params, "phase")

sim = compile(
    static, dim, params_est, params_opt, **{"optimize": "greedy", "argnum": 0}
)

def loss(params_est, params_opt):
    return sim.probabilities.cfim(params_est, params_opt).squeeze()

@jax.jit
def step(opt_state, params_est, params_opt):
    val, grad = jax.value_and_grad(loss, argnums=1)(params_est, params_opt)
    updates, opt_state = optimizer.update(grad, opt_state, params_opt)
    params_opt = optax.apply_updates(params_opt, updates)
    return params_opt, opt_state, val

# Run optimization
optimizer = optax.chain(optax.adam(learning_rate=1e-3), optax.scale(-1.0))
opt_state = optimizer.init(params_opt)

losses = []
for i in range(n_steps):
    params_opt, opt_state, val = step(opt_state, params_est, params_opt)
    losses.append(val)

circuit = eqx.combine(static, params_est, params_opt)

fig, ax = plt.subplots()
ax.plot(losses)
ax.set(
    xlabel="Optimization step",
    ylabel="Classical Fisher Information"
)

```