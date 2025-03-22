from dataclasses import dataclass
from typing import Any, Callable

import einops
import jax
import jax.numpy as jnp
import jax.random as jr
from beartype import beartype
from jaxtyping import PyTree


@dataclass
class SimulatorQuantumAmplitude:
    forward: Callable
    grad: Callable
    qfim: Callable

    def jit(self, device: jax.Device = None):
        return SimulatorQuantumAmplitude(
            forward=jax.jit(self.forward, device=device),
            grad=jax.jit(self.grad, device=device),
            qfim=jax.jit(self.qfim, static_argnames=("get",), device=device),
        )


def _quantum_fisher_information_matrix(
    get: Callable, amplitudes: PyTree, grads: PyTree
):
    _grads = get(grads)
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
    get: Callable,
    params: PyTree,
):
    amplitudes = _forward_amplitudes(params)
    grads = _grad_amplitudes(params)
    return _quantum_fisher_information_matrix(get, amplitudes, grads)


@dataclass
class SimulatorClassicalProbability:
    forward: Callable
    grad: Callable
    cfim: Callable

    @beartype
    def jit(self, device: jax.Device = None):
        return SimulatorClassicalProbability(
            forward=jax.jit(self.forward, device=device),
            grad=jax.jit(self.grad, device=device),
            cfim=jax.jit(self.cfim, static_argnames=("get",), device=device),
        )


def _classical_fisher_information_matrix(get: Callable, probs: PyTree, grads: PyTree):
    # return jnp.einsum(
    #     "i..., j... -> ij",
    #     get(grads),
    #     # get(grads) / (probs[None, ...] + 1e-14),
    #     jnp.nan_to_num(get(grads) / probs[None, ...], 0.0),
    # )
    return jnp.einsum(
        "i..., j..., ... -> ij",
        get(grads),
        get(grads),
        1 / (probs[None, ...] + 1e-14),
        # get(grads) / (probs[None, ...] + 1e-14),
        # jnp.nan_to_num(get(grads) / probs[None, ...], 0.0),
    )
    # return jnp.einsum("i..., j..., ... -> ij", get(grads), get(grads), 1 / probs)


def classical_fisher_information_matrix(
    _forward_prob: Callable,
    _grad_prob: Callable,
    get: Callable,
    params: PyTree,
):
    probs = _forward_prob(params)
    grads = _grad_prob(params)
    return _classical_fisher_information_matrix(get, probs, grads)


@dataclass
class Simulator:
    amplitudes: SimulatorQuantumAmplitude
    prob: SimulatorClassicalProbability
    path: Any
    info: str = None

    def jit(self, device: jax.Device = None):
        if not device:
            device = jax.devices()[0]

        return Simulator(
            amplitudes=self.amplitudes.jit(device=device),
            prob=self.prob.jit(device=device),
            path=self.path,
            info=self.info,
        )

    def sample(self, key: jr.PRNGKey, params: PyTree, shape: tuple[int, ...]):
        pr = self.prob.forward(params)
        idx = jnp.nonzero(pr)
        samples = einops.rearrange(
            jr.choice(key=key, a=jnp.stack(idx), p=pr[idx], shape=shape, axis=1),
            "s ... -> ... s",
        )
        return samples
