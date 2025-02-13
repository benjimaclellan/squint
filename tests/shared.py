#%%
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Int, ArrayLike

#%%
class Phase(eqx.Module):
    phi: Array

class TestShared(eqx.Module):
    shared: eqx.nn.Shared

    def __init__(self):
        a = jnp.array(1.0)
        b = jnp.array(1.0)
        
        # These two weights will now be tied together.
        where = lambda sh: sh[1]
        get = lambda sh: sh[0]
        self.shared = eqx.nn.Shared((a, b), where, get)

    def __call__(self):
        # Expand back out so we can evaluate these layers.
        a, b = self.shared()
        assert a is b  # same parameter!
        # Now go ahead and evaluate your language model.
        print(a, b)
        return 
    
    
model = TestShared()
model.shared()[0]

#%%
params, static = eqx.partition(model, eqx.is_inexact_array)
print(params)
params = eqx.tree_at(lambda params: params.shared.pytree[0], model, jnp.array(0.1))
params()


# %%
