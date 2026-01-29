"""
Tests for GPU precision of matmul and einsum operations.

These tests verify that JAX operations maintain proper precision on GPUs.
See: https://github.com/jax-ml/jax/issues/19444
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest


def has_gpu():
    """Check if a GPU device is available."""
    try:
        devices = jax.devices()
        return any(d.platform != "cpu" for d in devices)
    except Exception:
        return False


skip_no_gpu = pytest.mark.skipif(not has_gpu(), reason="No GPU available")


def get_device_info():
    """Get information about the current JAX device."""
    dev = jax.devices()[0]
    return "CPU" if dev.platform == "cpu" else dev.device_kind


def relative_residual(a, b):
    """Compute relative residual between two arrays."""
    nrm = max(jnp.linalg.norm(a.ravel()), jnp.linalg.norm(b.ravel()))
    if nrm == 0.0:
        return 0.0
    return jnp.linalg.norm((b - a).ravel()) / nrm


def matmul_associativity_error(A, B, x):
    """Compute associativity error: (A @ B) @ x vs A @ (B @ x)."""
    abx1 = (A @ B) @ x
    abx2 = A @ (B @ x)
    return relative_residual(abx1, abx2)


@skip_no_gpu
def test_einsum_matmul_consistency_complex64():
    """Test that einsum and matmul produce consistent results for complex64."""
    jax.config.update("jax_default_matmul_precision", "highest")

    key = jr.PRNGKey(0)
    subkeys = jr.split(key, 4)

    a = jr.normal(key=subkeys[0], shape=(10, 10), dtype=jnp.float64)
    b = jnp.exp(1j * a)
    c = jnp.exp(-1j * a)

    out_einsum = jnp.einsum("ab, bc -> ac", b, c)
    out_matmul = jnp.matmul(b, c)

    assert jnp.allclose(out_einsum, out_matmul), (
        f"einsum and matmul results differ on {get_device_info()}"
    )


@skip_no_gpu
def test_einsum_precision_float64():
    """Test einsum precision with float64 arrays on GPU."""
    jax.config.update("jax_default_matmul_precision", "highest")

    np.random.seed(42)
    A = np.random.rand(300, 300)
    B = np.random.rand(300, 300, 4)

    numpy_result = np.einsum("ab,caP->cbP", A, B)
    jax_result = jnp.einsum("ab,caP->cbP", jnp.array(A), jnp.array(B))

    # JAX result should be close to numpy double precision
    error = np.linalg.norm(jax_result - numpy_result)
    assert error < 1e-10, (
        f"JAX einsum deviates from NumPy by {error:.3e} on {get_device_info()}"
    )


@skip_no_gpu
@pytest.mark.parametrize("precision", ["default", "high", "highest"])
def test_matmul_associativity_float32(precision):
    """Test matmul associativity error at different precision levels."""
    if precision != "default":
        jax.config.update("jax_default_matmul_precision", precision)

    np.random.seed(1234)
    dtype = np.float32
    M, N = (8, 16)
    A = jnp.array(np.random.randn(M, N).astype(dtype=dtype))
    B = jnp.array(np.random.randn(N, M).astype(dtype=dtype))
    x = jnp.array(np.random.randn(M, 1).astype(dtype=dtype))

    error = matmul_associativity_error(A, B, x)

    # Error thresholds depend on precision level
    thresholds = {
        "default": 1e-4,  # Allow larger error for default precision
        "high": 1e-5,
        "highest": 1e-6,
    }

    assert error < thresholds[precision], (
        f"Matmul associativity error {error:.3e} exceeds threshold "
        f"for {precision} precision on {get_device_info()}"
    )
