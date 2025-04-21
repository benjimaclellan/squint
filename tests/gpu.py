# %%

import jax
import jax.numpy as jnp
import jax.random as jr

jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_default_matmul_precision', 'highest')

"""
https://github.com/jax-ml/jax/issues/19444
"""

# %%
key = jr.PRNGKey(0)
subkeys = jr.split(key, 4)

a = jr.normal(key=subkeys[0], shape=(10, 10), dtype=jnp.float64)
b = jnp.exp(1j * a)
c = jnp.exp(-1j * a)
# b = jr.normal(key=subkeys[1], shape=(100, 10), dtype=jnp.complex64)

# print(a)
print(a.device)

out = jnp.einsum("ab, bc -> ac", b, c)
out_ = jnp.matmul(b, c)
print(b * c)
print("Close", jnp.allclose(out, out_))

# %%
import jax
import jax.numpy as jnp
import numpy as np

A = np.random.rand(300, 300)
B = np.random.rand(300, 300, 4)

numpy_result_double = np.einsum("ab,caP->cbP", A, B)
numpy_result_single = np.einsum(
    "ab,caP->cbP", A.astype(np.float32), B.astype(np.float32)
)

jax_result = jnp.einsum("ab,caP->cbP", jnp.array(A), jnp.array(B))

print(np.linalg.norm(numpy_result_double - numpy_result_single))
print(np.linalg.norm(jax_result - numpy_result_single))

# %%
import jax
import jax.numpy as jnp
import numpy as np


def dev_info():
    dev = jax.devices()[0]
    info = "CPU" if dev.platform == "cpu" else dev.device_kind
    return info


def relres(ax, b):
    nrm = max(jnp.linalg.norm(ax.ravel()), jnp.linalg.norm(b.ravel()))
    if nrm == 0.0:
        return 0.0
    return jnp.linalg.norm((b - ax).ravel()) / nrm


def mx_mul_assoc_error(A, B, x):
    abx1 = (A @ B) @ x
    abx2 = A @ (B @ x)
    return relres(abx1, abx2)


np.random.seed(1234)
dtype = np.float32
M, N = (8, 16)
A = jnp.array(np.random.randn(M, N).astype(dtype=dtype))
B = jnp.array(np.random.randn(N, M).astype(dtype=dtype))
x = jnp.array(np.random.randn(M, 1).astype(dtype=dtype))

print(f"Running test on device {dev_info()}. Matrix mult. assoc. error at:")
print(f"    default matmul precision: {mx_mul_assoc_error(A, B, x):.3e}")
jax.config.update("jax_default_matmul_precision", "high")
print(f"    high matmul precision:    {mx_mul_assoc_error(A, B, x):.3e}")
jax.config.update("jax_default_matmul_precision", "highest")
print(f"    highest matmul precision: {mx_mul_assoc_error(A, B, x):.3e}")
