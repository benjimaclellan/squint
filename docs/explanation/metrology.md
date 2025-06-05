## Quantum metrology

**Quantum metrology** is the science of using quantum mechanical phenomena to make precise measurements. The key insight is that quantum resources like entanglement and squeezing can provide measurement precision beyond what's possible with classical methods.

Every quantum sensing protocol follows a four-step process:

1. **State Preparation**: Create a probe state, $|\psi_0\rangle$
2. **Parameter Encoding**: Let the unknown parameter $\varphi$ interact with the probe: $|\psi(\varphi)\rangle = U(\varphi)|\psi_0\rangle$
3. **Measurement**: Perform a measurement yielding outcome probabilities, $p(s_i|\varphi)$
4. **Estimation**: Estimate $\bar{\varphi}$ from the measurement data

#### Classical vs quantum Limits

The precision of any unbiased estimator is bounded by the Cramér-Rao bound:

$$\Delta^2\bar{\varphi} \geq \frac{1}{m \mathcal{I}_\varphi}$$

where $m$ is the number of measurements and $\mathcal{I}_\varphi$ is the Fisher Information.

- **Classical sensors**: $\mathcal{I}_\varphi \propto n$ (Standard Quantum Limit)
- **Quantum sensors**: $\mathcal{I}_\varphi \propto n^2$ (Heisenberg Limit)

This quadratic improvement means quantum sensors can achieve the same precision with exponentially fewer resources.

#### Fisher Information

The **Fisher Information** quantifies how much information measurement data contains about an unknown parameter. 

The Quantum Fisher Information (QFI) represents the maximum information extractable from a quantum state, achieved by the optimal measurement:

$$ \mathcal{I}_\varphi^{(Q)} = 4(\langle\partial_\varphi\psi|\partial_\varphi\psi\rangle - |\langle\psi|\partial_\varphi\psi\rangle|^2) $$

For mixed states,

$$\mathcal{I}_\varphi^{(Q)} = \sum_{i,j} \frac{2\text{Re}(\langle\lambda_i|\partial_\varphi\rho|\lambda_j\rangle)}{\lambda_i + \lambda_j}$$

The CFI quantifies the information in actual measurement outcomes:

$$\mathcal{I}_\varphi^{(C)} = \sum_i \frac{(\partial_\varphi p(s_i|\varphi))^2}{p(s_i|\varphi)}$$

**Key relationship**: $\mathcal{I}_\varphi^{(C)} \leq \mathcal{I}_\varphi^{(Q)}$ always, with equality achieved by optimal measurements.

### Why Variational Quantum Sensing?

Traditional quantum sensing protocols assume:

- Perfect quantum gates
- Ideal measurements  
- Specific quantum states (like GHZ)
- No noise or decoherence

**Variational Quantum Sensing (VQS)** addresses real-world constraints by:

1. **Parameterizing** all protocol components (state preparation, measurements)
2. **Optimizing** parameters to maximize Fisher Information
3. **Adapting** to hardware constraints and noise
4. **Learning** optimal strategies from data

This approach enables:

- Hardware-efficient protocols
- Noise resilience
- Automatic protocol discovery
- End-to-end optimization


`squint` leverages JAX's automatic differentiation to compute, e.g., gradients of amplitudes and probability distributions, and to compute the Fisher Inforamtion itself.
For a parameterized quantum state $|\psi(\varphi)\rangle$, we can compute:

```python
# State and its gradient
psi = circuit.amplitudes.forward(params)      # |ψ(θ)⟩
dpsi = circuit.amplitudes.grad(params)        # ∂|ψ(θ)⟩/∂θ

# Probabilities and their gradients  
p = circuit.probabilities.forward(params)     # p(s|θ)
dp = circuit.probabilities.grad(params)       # ∂p(s|θ)/∂θ

# Fisher Information Matrices
qfim = circuit.amplitudes.qfim(params)        # Quantum FIM
cfim = circuit.probabilities.cfim(params)     # Classical FIM
```