
## Optimizing a sensor

This guide shows you how to use Squint's optimization capabilities to design optimal sensing protocols.

```python
import optax
from squint.circuit import Circuit
from squint.ops.dv import *

def create_variational_sensor(n_qubits, n_layers):
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
    for i in range(n_qubits):
        circuit.add(RZGate(wires=(i,), phi=0.0), f"phase_{i}")
    
    # Measurement layer
    for i in range(n_qubits):
        circuit.add(RXGate(wires=(i,), phi=0.0), f"meas_x_{i}")
        circuit.add(RYGate(wires=(i,), phi=0.0), f"meas_y_{i}")
    
    return circuit
```

### Optimization loop

```python
def optimize_sensor(circuit, n_qubits, n_layers, learning_rate=0.01, n_steps=1000):
    """Optimize the variational quantum sensor using Fisher Information."""
    
    # Separate phase and variational parameters
    phase_keys = [f"phase_{i}" for i in range(n_qubits)]
    var_keys = [k for k in circuit.get_parameter_keys() if k not in phase_keys]
    
    # Compile circuit
    dim = 2
    params, static = partition_op(circuit, phase_keys + var_keys)
    sim = circuit.compile(static, dim, params).jit()
    
    # Initialize parameters
    key = jax.random.PRNGKey(42)
    var_params = {}
    for k in var_keys:
        var_params[k] = {"phi": jax.random.uniform(key, (), minval=0, maxval=2*jnp.pi)}
        key, _ = jax.random.split(key)
    
    # Fixed phase for optimization
    phase_params = {k: {"phi": jnp.pi/4} for k in phase_keys}
    
    # Optimization setup
    optimizer = optax.adam(learning_rate)
    
    def loss_fn(var_params):
        """Loss function: negative Classical Fisher Information."""
        all_params = {**phase_params, **var_params}
        cfi = sim.probabilities.cfim(all_params)
        return -jnp.sum(cfi)  # Maximize CFI
    
    opt_state = optimizer.init(var_params)
    
    # Optimization history
    loss_history = []
    cfi_history = []
    
    @jax.jit
    def update_step(var_params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(var_params)
        updates, opt_state = optimizer.update(grads, opt_state)
        var_params = optax.apply_updates(var_params, updates)
        return var_params, opt_state, loss
    
    # Optimization loop
    for step in range(n_steps):
        var_params, opt_state, loss = update_step(var_params, opt_state)
        
        if step % 100 == 0:
            all_params = {**phase_params, **var_params}
            cfi = jnp.sum(sim.probabilities.cfim(all_params))
            
            loss_history.append(float(loss))
            cfi_history.append(float(cfi))
            
            print(f"Step {step}: Loss = {loss:.4f}, CFI = {cfi:.4f}")
    
    return var_params, loss_history, cfi_history

# Run optimization
n_qubits = 4
n_layers = 3
circuit = create_variational_sensor(n_qubits, n_layers)
optimal_params, losses, cfis = optimize_sensor(circuit, n_qubits, n_layers)

# Plot optimization progress
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Loss (Negative CFI)')
plt.xlabel('Optimization Steps (×100)')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(cfis)
plt.axhline(y=n_qubits**2, color='r', linestyle='--', label='Heisenberg Limit')
plt.axhline(y=n_qubits, color='g', linestyle='--', label='Standard Quantum Limit')
plt.title('Classical Fisher Information')
plt.xlabel('Optimization Steps (×100)')
plt.ylabel('CFI')
plt.legend()

plt.tight_layout()
plt.show()
```