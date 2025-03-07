# %%
import jax

# %%

def f(x):
    return x**2


f(10)
# fx = jax.jit(f, device=jax.devices('cpu')[0])

fx = jax.jit(f, device=None)


arr = fx(10.0)
print(arr.device)


#%%