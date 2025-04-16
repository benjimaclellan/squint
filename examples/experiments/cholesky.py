# %%
import jax.random as jr
import jax.scipy as jsp
import optax

key = jr.PRNGKey(0)

tril = jnp.tril(jr.normal(key, shape=(2, 2)))
herm = tril @ tril.conj().T
u = jsp.linalg.expm(1j * herm)
print(u)
print(u.conj().T @ u)

# %%
n = 2
params = jr.normal(key, shape=(2**n, 2**n))


def unitary(params):
    tril = jnp.tril(params)
    herm = tril @ tril.conj().T
    u = jsp.linalg.expm(1j * herm)
    return u


u = unitary(params)

z = jnp.array([[1, 0], [0, -1]])
x = jnp.array([[0, 1], [1, 0]])
eye = jnp.array([[1, 0], [0, 1]])
ket0, ket1 = jnp.array([[1, 0]]).T, jnp.array([[0, 1]]).T
cnot = jnp.kron(ket0 @ ket0.T, eye) + jnp.kron(ket1 @ ket1.T, x)
had = jnp.array([[1, -1], [-1, 1]]) / jnp.sqrt(2)
v = cnot
print(u.T.conj() @ u)
print(z.T.conj() @ z)
1 - jnp.abs(jnp.trace(v.T.conj() @ u)) / 2**n


# %%
def loss_fn(params):
    u = unitary(params)
    return 1 - jnp.abs(jnp.trace(v.T.conj() @ u)) / 2**n


start_learning_rate = 1e-1
optimizer = optax.adam(start_learning_rate)

# Initialize parameters of the model + optimizer.
opt_state = optimizer.init(params)

for i in range(1000):
    val, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    #   print(params)
    print(i, val, jnp.max(grads))

# %%
u = unitary(params)
print(u)


# %%
