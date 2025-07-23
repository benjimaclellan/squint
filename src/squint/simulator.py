# Copyright 2024-2025 Benjamin MacLellan

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any, Callable

import einops
import jax
import jax.numpy as jnp
import jax.random as jr
from beartype import beartype
from jaxtyping import Array, PyTree

__all__ = ["SimulatorQuantumAmplitudes", "SimulatorClassicalProbabilities", "Simulator"]


@dataclass
class SimulatorQuantumAmplitudes:
    """
    Simulator object which computes quantities related to the quantum probability amplitudes,
    including forward pass, gradient computation,
    and quantum Fisher information matrix calculation.
    
    Attributes:
        forward (Callable): Function to compute quantum amplitudes.
        grad (Callable): Function to compute gradients of quantum amplitudes.
        qfim (Callable): Function to compute the quantum Fisher information matrix.
    """

    forward: Callable
    grad: Callable
    qfim: Callable

    def jit(self, device: jax.Device = None):
        """
        JIT (just-in-time) compile the simulator methods.
        Args:
            device (jax.Device, optional): Device to compile the methods on. Defaults to None, which uses the first available device.
        """
        return SimulatorQuantumAmplitudes(
            forward=jax.jit(self.forward, device=device),
            grad=jax.jit(self.grad, device=device),
            qfim=jax.jit(self.qfim, device=device),
            # qfim=jax.jit(self.qfim, static_argnames=("get",), device=device),
        )


def _quantum_fisher_information_matrix(
    # get: Callable,
    amplitudes: Array,
    grads: Array,
):
    """
    Computes the quantum Fisher information matrix from the already computed arrays representing
    the probability amplitudes and their gradients.
    
    Args:
        amplitudes (Array): Quantum amplitudes.
        grads (Array): Gradients of the quantum amplitudes.

    Returns:
        qfim (jnp.ndarray): Quantum Fisher information matrix.
    """
    _grads = grads
    _grads_conj = jnp.conjugate(_grads)
    return 4 * jnp.real(
        jnp.real(jnp.einsum("i..., j... -> ij", _grads_conj, _grads))
        + jnp.einsum(
            "i,j->ij",
            jnp.einsum("i..., ... -> i", _grads_conj, amplitudes),
            jnp.einsum("j..., ... -> j", _grads_conj, amplitudes),
        )
    )


def quantum_fisher_information_matrix(
    _forward_amplitudes: Callable,
    _grad_amplitudes: Callable,
    # get: Callable,
    *params: PyTree,
):
    """
    Performs the forward pass to compute quantum amplitudes and their gradients,
    and then calculates the quantum Fisher information matrix.
    Args:
        _forward_amplitudes (Callable): Function to compute quantum amplitudes.
        _grad_amplitudes (Callable): Function to compute gradients of quantum amplitudes.
        *params (list[PyTree]): Parameters for the quantum circuit, partitioned via `eqx.partition`.
            The argnum is already defined in the callables
    Returns:
        qfim (jnp.ndarray): Quantum Fisher information matrix."""
    amplitudes = _forward_amplitudes(*params)
    grads, _ = jax.tree.flatten(_grad_amplitudes(*params))
    grads = jnp.stack(grads, axis=0)
    return _quantum_fisher_information_matrix(amplitudes, grads)


@dataclass
class SimulatorClassicalProbabilities:
    """
    Simulator object which computes quantities related to the classical probabilities,
    including forward pass, gradient computation,
    and classical Fisher information matrix calculation.
    
    Attributes:
        forward (Callable): Function to compute classical probabilities.
        grad (Callable): Function to compute gradients of classical probabilities.
        cfim (Callable): Function to compute the classical Fisher information matrix.
    """

    forward: Callable
    grad: Callable
    cfim: Callable

    @beartype
    def jit(self, device: jax.Device = None):
        """
        JIT (just-in-time) compile the simulator methods.
        Args:
            device (jax.Device, optional): Device to compile the methods on. Defaults to None, which uses the first available device.
        """
        return SimulatorClassicalProbabilities(
            forward=jax.jit(self.forward, device=device),
            grad=jax.jit(self.grad, device=device),
            cfim=jax.jit(self.cfim, device=device),
            # cfim=jax.jit(self.cfim, static_argnames=("get",), device=device),
        )


def _classical_fisher_information_matrix(
    probs: Array,
    grads: Array,
):
    """
    Computes the classical Fisher information matrix from the already computed arrays representing
    the probabilities and their gradients.
    Args:
        probs (Array): Classical probabilities.
        grads (Array): Gradients of the classical probabilities.
    Returns:
        cfim (jnp.ndarray): Classical Fisher information matrix.
    """

    return jnp.einsum(
        "i..., j..., ... -> ij",
        grads,
        grads,
        1
        / (probs[None, ...] + 1e-14),  # add a small constant to avoid division by zero
    )


def classical_fisher_information_matrix(
    _forward_prob: Callable,
    _grad_prob: Callable,
    # get: Callable,
    *params: PyTree,
):
    """
    Performs the forward pass to compute classical probabilities and their gradients,
    and then calculates the classical Fisher information matrix.
    Args:
        _forward_prob (Callable): Function to compute classical probabilities.
        _grad_prob (Callable): Function to compute gradients of classical probabilities.
        *params (list[PyTree]): Parameters for the quantum circuit, partitioned via `eqx.partition`.
            The argnum is already defined in the callables
    Returns:
        cfim (jnp.ndarray): Classical Fisher information matrix.
    """
    probs = _forward_prob(*params)
    grads, _ = jax.tree.flatten(_grad_prob(*params))
    grads = jnp.stack(grads, axis=0)
    return _classical_fisher_information_matrix(probs, grads)


@dataclass
class Simulator:
    """
    Simulator for quantum circuits, providing callable methods for computing
    forwar, backward, and Fisher Information matrix calculations on the
    quantum amplitudes and classical probabilities, given a set of parameters PyTrees

    Attributes:
        amplitudes (SimulatorQuantumAmplitudes): Object for quantum amplitudes computations.
        probabilities (SimulatorClassicalProbabilities): Object for classical probabilities computations.
        path (Any): Path to the simulator, can be used for saving/loading.
        info (str, optional): Additional information about the simulator.
    """

    amplitudes: SimulatorQuantumAmplitudes
    probabilities: SimulatorClassicalProbabilities
    path: Any
    info: str = None

    def jit(self, device: jax.Device = None):
        """
        JIT (just-in-time) compile the simulator methods.
        Args:
            device (jax.Device, optional): Device to compile the methods on. Defaults to None, which uses the first available device.
        """
        if not device:
            device = jax.devices()[0]

        return Simulator(
            amplitudes=self.amplitudes.jit(device=device),
            probabilities=self.probabilities.jit(device=device),
            path=self.path,
            info=self.info,
        )

    def sample(self, key: jr.PRNGKey, params: PyTree, shape: tuple[int, ...]):
        """
        Sample from the quantum circuit using the provided parameters and a random key.
        Args:
            key (jr.PRNGKey): Random key for sampling.
            params (PyTree): Parameters for the quantum circuit, partitioned via `eqx.partition`.
            shape (tuple[int, ...]): Shape of the output samples.
        Returns:
            samples (jnp.ndarray): Samples drawn from the quantum circuit.
        """
        pr = self.probabilities.forward(params)
        idx = jnp.nonzero(pr)
        samples = einops.rearrange(
            jr.choice(key=key, a=jnp.stack(idx), p=pr[idx], shape=shape, axis=1),
            "s ... -> ... s",
        )
        return samples
