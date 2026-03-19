
import functools
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Sequence, Union

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import paramax
from beartype import beartype
from beartype.typing import Type
from jaxtyping import Array, PyTree
from opt_einsum.parser import get_symbol
from ordered_set import OrderedSet

def qfim(
    psi: Array,
    dspi: Array,
):
    """
    Computes the quantum Fisher information matrix from the already computed arrays representing
    the probability amplitudes and their gradients.

    Args:
        psi (Array): Quantum amplitudes.
        dpsi (Array): Gradients of the quantum amplitudes.

    Returns:
        qfim (jnp.ndarray): Quantum Fisher information matrix.
    """
    dpsi_conj = jnp.conjugate(dspi)
    return 4 * jnp.real(
        jnp.real(jnp.einsum("i..., j... -> ij", dpsi_conj, dspi))
        + jnp.einsum(
            "i,j->ij",
            jnp.einsum("i..., ... -> i", dpsi_conj, psi),
            jnp.einsum("j..., ... -> j", dpsi_conj, psi),
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
    return qfim(amplitudes, grads)




def cfim(
    p: Array,
    dp: Array,
):
    """
    Computes the classical Fisher information matrix from the already computed arrays representing
    the probabilities and their gradients.
    Args:
        p (Array): Classical probabilities.
        dp (Array): Gradients of the classical probabilities.
    Returns:
        cfim (jnp.ndarray): Classical Fisher information matrix.
    """

    return jnp.einsum(
        "i..., j..., ... -> ij",
        dp,
        dp,
        1
        / (p[None, ...] + 1e-14),  # add a small constant to avoid division by zero
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
    return cfim(probs, grads)
