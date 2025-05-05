# %%
import functools
import pathlib
from typing import ClassVar
import os 
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
import tqdm 
import paramax
import treescope
import ultraplot as uplt
from beartype import beartype
from equinox.nn import MLP
from flowjax.bijections.bijection import AbstractBijection
from flowjax.distributions import StandardNormal, Transformed
from jax.scipy.linalg import solve_triangular
from jaxtyping import ArrayLike, PRNGKeyArray
import dotenv 
from squint import hdfdict

dotenv.load_dotenv()


class AbstractEstimator(eqx.Module):
    @beartype
    def __init__(self, wires: tuple[int]):
        """ """
        pass


class PermutationInvariantNeuralNetwork(eqx.Module):
    f: MLP
    g: MLP

    @beartype
    def __init__(
        self,
        f: eqx.Module,
        g: eqx.Module,
    ):
        self.f = f
        self.g = g
        pass

    def __call__(self, x):
        h1 = jax.vmap(self.f)(x)
        h2 = h1.sum(axis=0)
        # h2 = h1.mean(axis=0)
        h3 = self.g(h2)
        return h3

    def batch_cumsum(self, x):
        h1 = jax.vmap(self.f)(x)
        h2 = jnp.cumsum(h1, axis=0)
        # h2 = h1.mean(axis=0)
        h3 = jax.vmap(self.g)(h2)
        return h3
    
    
class BayesFlowEstimator(AbstractEstimator):
    pinn_loc: PermutationInvariantNeuralNetwork
    pinn_scale: PermutationInvariantNeuralNetwork

    n_params: int
    n_wires: int

    @beartype
    def __init__(
        self,
        pinn_loc: PermutationInvariantNeuralNetwork,
        pinn_scale: PermutationInvariantNeuralNetwork,
        n_params: int,
        n_wires: int,
    ):
        self.pinn_loc = pinn_loc
        self.pinn_scale = pinn_scale
        self.n_params = n_params
        self.n_wires = n_wires

    @beartype
    @classmethod
    def init_bayesflow(
        cls,
        key: PRNGKeyArray,
        n_params: int,
        n_wires: int,
        latent_dim: int = 4,
        kwargs_loc: dict = {},
        kwargs_scale: dict = {},
    ):
        subkeys = jr.split(key, 4)
        f_loc = MLP(
            key=subkeys[0],
            in_size=n_wires,
            out_size=latent_dim,
            width_size=kwargs_loc.get("width_size", 4),
            depth=kwargs_loc.get("depth", 3),
            activation=jax.nn.elu,
            use_final_bias=False,
            use_bias=False,
        )
        g_loc = MLP(
            key=subkeys[1],
            in_size=latent_dim,
            out_size=n_params,
            width_size=kwargs_loc.get("width_size", 4),
            depth=kwargs_loc.get("depth", 3),
            activation=jax.nn.elu,
            use_final_bias=False,
            use_bias=False,
        )

        f_scale = MLP(
            key=subkeys[2],
            in_size=n_wires,
            out_size=latent_dim,
            width_size=kwargs_scale.get("width_size", 4),
            depth=kwargs_scale.get("depth", 3),
            activation=jax.nn.elu,
            use_final_bias=False,
            use_bias=True,
        )
        g_scale = MLP(
            key=subkeys[3],
            in_size=latent_dim,
            out_size=n_params**2,
            width_size=kwargs_scale.get("width_size", 4),
            depth=kwargs_scale.get("depth", 3),
            activation=jax.nn.elu,
            # final_activation=jax.nn.softplus,
            use_final_bias=False,
            use_bias=True,
        )
        pinn_loc = PermutationInvariantNeuralNetwork(f=f_loc, g=g_loc)
        pinn_scale = PermutationInvariantNeuralNetwork(f=f_scale, g=g_scale)

        return cls(
            pinn_loc=pinn_loc, pinn_scale=pinn_scale, n_params=n_params, n_wires=n_wires
        )

    def __call__(self, x):
        loc = self.pinn_loc(x)
        scale = self.pinn_scale(x)
        # return loc, scale.reshape(self.n_params, self.n_params)
        # return loc, 2 ** scale.reshape(self.n_params, self.n_params)
        return loc, jax.nn.softplus(scale.reshape(self.n_params, self.n_params))
    
    def batch_cumsum(self, x):
        loc = self.pinn_loc.batch_cumsum(x)
        scale = self.pinn_scale.batch_cumsum(x)
        # return loc, scale.reshape(scale.shape[0], self.n_params, self.n_params)
        # return loc, 2 ** scale.reshape(scale.shape[0], self.n_params, self.n_params)
        return loc, jax.nn.softplus(scale.reshape(scale.shape[0], self.n_params, self.n_params))
        # return loc, scale

# %%
n_wires = 4
datapath = pathlib.Path(os.getenv("DATAPATH")).joinpath("ghz.h5")
dataset = hdfdict.load(datapath)
data = dataset[f"wires={n_wires}"]["d=2"]
shots, phis = data["shots"], data["phis"]
shots = 2 * shots - 1
dataset.close()

# shots = jnp.ones_like(shots) * phis[:, None]

# %%
n_params = 1

summary = BayesFlowEstimator.init_bayesflow(
    n_wires=n_wires,
    n_params=n_params,
    latent_dim=16,
    key=jr.PRNGKey(122345),
    kwargs_loc={"width_size": 8, "depth": 4},
    kwargs_scale={"width_size": 8, "depth": 4},
)


loc, scale = summary.batch_cumsum(shots[0, 0:1000, :])
print(loc.shape, scale.shape)

loc, scale = summary(shots[0, 0:1000, :])
print(loc.shape, scale.shape)
print(loc, scale)


#%%
params, static = eqx.partition(summary, eqx.is_array)

lr = 1e-3
optimizer = optax.chain(optax.adam(lr))
opt_state = optimizer.init(params)

conditional = shots
x = phis

def shuffle(key, m_phis=20, m_shots=35, n_phis=256, n_shots=4096):
    subkeys = jr.split(key, 2)
    idx_phis = jr.randint(subkeys[0], shape=(m_phis,), minval=0, maxval=n_phis)
    idx_shots = jr.randint(subkeys[1], shape=(m_shots,), minval=0, maxval=n_shots)
    return idx_phis, idx_shots


def loss_fn(key, params, static, x, conditional):
    flow = paramax.unwrap(eqx.combine(params, static))
    
    subkeys = jr.split(key, 3)
    idx_phis, idx_shots = shuffle(subkeys[0])
    
    loc, scale = jax.vmap(flow.batch_cumsum)(jr.permutation(subkeys[1], conditional[idx_phis], axis=1))
    # scale = jax.nn.softplus(scale)[..., 0]
    scale = scale[..., 0]
    
    # sigma = 1 / jnp.sqrt(scale)
    sigma = scale
    # nll = -jnp.log(scale) - 0.5 * jnp.log(2 * jnp.pi) - 0.5 * ((x[idx_phis, None, :] - loc) / scale)**2
    nll = -jnp.log(sigma) - 0.5 * jnp.log(2 * jnp.pi) - 0.5 * ((x[idx_phis, None, :] - loc) / sigma )**2
    return -nll[:, idx_shots, :].mean()
    

@jax.jit
def step(key, opt_state, params, x, conditional):
    key, *subkeys = jr.split(key, 3)    
    loss, grad = jax.value_and_grad(
        functools.partial(loss_fn, static=static), argnums=1
    )(subkeys[0], params, x=x, conditional=conditional)
    
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, grad, loss

# %%
"""
For a training loop we have:
shots: [n_batch, n_shots, n_wires]
phis: [n_batch, n_params]

With few shots, we have more batches that we can train on (e.g., training on single shots, we have n_shots examples)
Options:
1. Create a fixed number of reshaped arrays by copying the data. Likely the fastest, but uses more memory.
2. Use cumsum and compute the full dataset always, post-select some to include in the loss function.
3. 
"""

key = jr.PRNGKey(1234)
pbar = tqdm.tqdm(range(1, 1000))
indices = jnp.arange(shots.shape[1])

losses = []
for i in pbar:
    key, subkey = jr.split(key, 2)
    params, opt_state, grad, loss = step(subkey, opt_state, params, phis, shots)
    pbar.set_postfix(loss=float(loss), step=i)
    losses.append(loss)
    
flow = paramax.unwrap(eqx.combine(params, static))

fig, ax = uplt.subplot()
ax.plot(losses)
ax.set(xlabel='Steps', ylabel="Loss")
fig.save("loss.png")

#%%
locs, scales = jax.vmap(flow.batch_cumsum)(shots)

#%%
samples = jr.normal(key, shape=(10000,))

m = 4000

fig, axs = uplt.subplots(ncols=1, nrows=2, figsize=(10, 5), sharex=True)
loc, scale =  locs[0, m].squeeze(), scales[0, m].squeeze()
axs[0].hist(loc + scale * samples, bins=100)

loc, scale =  locs[200, m].squeeze(), scales[200, m].squeeze()
axs[1].hist(loc + scale * samples, bins=100)
fig.save("samples.png")

fig, axs = uplt.subplots(ncols=1, nrows=2, figsize=(10, 5), sharex=False)

axs[0].plot(phis.squeeze(), locs[:, m, :].squeeze())
axs[0].plot(phis.squeeze(), phis.squeeze())
axs[1].plot(phis.squeeze(), scales[:, m, :].squeeze())
fig.save("loc_scale.png")

# axs[1].plot(jnp.log(1 / (n_wires**2 * jnp.arange(1, shots.shape[1]))))
# axs[0].plot(phis.squeeze(), scales.squeeze())

# %%
fig, axs = uplt.subplots(ncols=1, nrows=2, figsize=(10, 5), sharex=False)

ms = jnp.arange(1, shots.shape[1]+1)
axs[0].plot(ms, locs[100, :, :].squeeze())
axs[0].plot(ms, locs[200, :, :].squeeze())

axs[1].plot(ms, scales[100, :, :].squeeze())
axs[1].plot(ms, scales[200, :, :].squeeze())
fig.save("scales_ms.png")
