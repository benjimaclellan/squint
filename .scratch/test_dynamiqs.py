# %%
import dynamiqs as dq
import jax.numpy as jnp  # the JAX version of numpy

# %%
# parameters
n = 10  # number of Fock states
alpha = jnp.array(1j * 1.0)  #
T = jnp.pi  # time of evolution

# operators
a = dq.destroy(n)  # annihilation operator
# H = alpha * a **2 - jnp.conjugate(alpha) * dq.dag(a) **2
H = alpha * a.dag() - alpha.conj() * a
# H = (0.0 * a.dag() @ a + (alpha * a.dag() - alpha.conj() * a))

# initial state and save time
# psi0 = dq.coherent(n, 1.0)  # coherent state
psi0 = dq.fock(n, 4)  # coherent state
t_save = jnp.linspace(0, T, 500)  # save times

dq.plot.wigner(psi0)
# dq.plot.wigner((H).expm() @ psi0)
# dq.plot.wigner(dq.coherent(n, alpha))

# %%
# solve Schrodinger equation
result = dq.sesolve(H, psi0, t_save)
print(result)

# dq.plot.wigner(psi0)
# %%
dq.plot.wigner_mosaic(result.states, cross=True)

# %%
dq.plot.wigner_gif(result.states, gif_duration=2, fps=25)
