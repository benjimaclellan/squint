#%%
import jax
import jax.numpy as jnp

def f(x):
    return jnp.exp(jnp.cos(x))**2

y = f(x=1.0)
print(y)

grad_f = jax.grad(f)
print(grad_f(1.0))

#%%
jax.make_jaxpr(grad_f)(1.0)


#%% weighted coin example

def weighted_coin(theta):
    return jnp.array([theta, 1 - theta])


theta = 0.05
prob = weighted_coin(theta)
prob_grad = jax.jacrev(weighted_coin)(theta)
print(jnp.sum(jnp.abs(prob_grad) ** 2 / prob))