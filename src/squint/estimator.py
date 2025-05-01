# %%
import functools
import pathlib
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
import paramax
import treescope
import ultraplot as uplt
from beartype import beartype
from equinox.nn import MLP
from flowjax.bijections.bijection import AbstractBijection
from flowjax.distributions import StandardNormal, Transformed
from jax.scipy.linalg import solve_triangular
from jaxtyping import ArrayLike, PRNGKeyArray

from squint import hdfdict


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
        h3 = self.g(h2)
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
            activation=jax.nn.relu,
        )
        g_loc = MLP(
            key=subkeys[1],
            in_size=latent_dim,
            out_size=n_params,
            width_size=kwargs_loc.get("width_size", 4),
            depth=kwargs_loc.get("depth", 3),
            activation=jax.nn.relu,
        )

        f_scale = MLP(
            key=subkeys[2],
            in_size=n_wires,
            out_size=latent_dim,
            width_size=kwargs_scale.get("width_size", 4),
            depth=kwargs_scale.get("depth", 3),
            activation=jax.nn.relu,
        )
        g_scale = MLP(
            key=subkeys[3],
            in_size=latent_dim,
            out_size=n_params**2,
            width_size=kwargs_scale.get("width_size", 4),
            depth=kwargs_scale.get("depth", 3),
            activation=jax.nn.relu,
        )
        pinn_loc = PermutationInvariantNeuralNetwork(f=f_loc, g=g_loc)
        pinn_scale = PermutationInvariantNeuralNetwork(f=f_scale, g=g_scale)

        return cls(
            pinn_loc=pinn_loc, pinn_scale=pinn_scale, n_params=n_params, n_wires=n_wires
        )

    def __call__(self, x):
        loc = self.pinn_loc(x)
        scale = self.pinn_scale(x)
        return loc, scale.reshape(self.n_params, self.n_params)


# %%
datapath = pathlib.Path(
    r"/Users/benjamin/Desktop/1 - Projects/Quantum Intelligence Lab/repos/squint/datasets/ghz.h5"
)
dataset = hdfdict.load(datapath)
data = dataset["wires=4"]["d=2"]
shots, phis = data["shots"], data["phis"]


class TriangularAffine(AbstractBijection):
    r"""A triangular affine transformation.

    Transformation has the form :math:`Ax + b`, where :math:`A` is a lower or upper
    triangular matrix, and :math:`b` is the bias vector. We assume the diagonal
    entries are positive, and constrain the values using softplus. Other
    parameterizations can be achieved by e.g. replacing ``self.triangular``
    after construction.

    Args:
        loc: Location parameter. If this is scalar, it is broadcast to the dimension
            inferred from arr.
        arr: Triangular matrix.
        lower: Whether the mask should select the lower or upper
            triangular matrix (other elements ignored). Defaults to True (lower).
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    summary_network: eqx.Module
    lower: bool

    def __init__(
        self,
        dim: int,
        summary_network: eqx.Module,
        *,
        lower: bool = True,
    ):
        self.summary_network = summary_network
        self.lower = lower
        self.shape = (dim,)

    # @partial(jnp.vectorize, signature="(d,d)->(d,d)")
    def _to_triangular(self, arr):
        tri = jnp.tril(arr) if self.lower else jnp.triu(arr)
        # return jnp.fill_diagonal(tri, softplus(jnp.diag(tri)), inplace=False)
        return jnp.fill_diagonal(tri, jax.nn.softplus(jnp.diag(tri)), inplace=False)

    def loc_and_scale(self, condition):
        loc, scale = self.summary_network(condition)
        return loc, self._to_triangular(scale)

    def transform_and_log_det(self, x, condition: ArrayLike = None):
        loc, triangular = self.loc_and_scale(condition)
        y = triangular @ x + loc
        return y, jnp.log(jnp.abs(jnp.diag(triangular))).sum()

    def inverse_and_log_det(self, y, condition: ArrayLike = None):
        loc, triangular = self.loc_and_scale(condition)
        x = solve_triangular(triangular, y - loc, lower=self.lower)
        return x, -jnp.log(jnp.abs(jnp.diag(triangular))).sum()


# %%
n_params = 1
n_wires = 4

flow = BayesFlowEstimator.init_bayesflow(
    n_wires=n_wires,
    n_params=n_params,
    latent_dim=8,
    key=jr.PRNGKey(12345),
    kwargs_loc={"width_size": 4, "depth": 3},
    kwargs_scale={"width_size": 4, "depth": 3},
)

loc, scale = flow(shots[0])

# %%
bij = TriangularAffine(
    dim=n_params,
    summary_network=flow,
    lower=True,
)
z = bij.loc_and_scale(condition=shots[0, 0:10, :])
# bij._to_triangular(jnp.array([[-10, 1.0], [1.0, 22]]))
bij.transform_and_log_det(jnp.ones([n_params]), condition=shots[0, 0:10, :])

# %%
dist = StandardNormal(shape=(n_params,))
dist.sample(key=jr.key(1234), sample_shape=(1000,))


dist1 = Transformed(dist, bij)
samples = dist1.sample(
    key=jr.key(1234), sample_shape=(10000,), condition=shots[3, 0:100, :]
)
samples
flow = dist1

# %%
# plt.hist2d(x=samples[:, 0], y=samples[:, 1], bins=100)
plt.hist(samples, bins=100)
# plt.gca().set_aspect("equal")


# %%


params, static = eqx.partition(flow, eqx.is_array)


def log_prob(flow, x, condition):
    return -flow.log_prob(x, condition)  # .mean()


def batch_loss_fn(params, static, x, condition):
    flow = paramax.unwrap(eqx.combine(params, static))
    return -jax.vmap(log_prob, in_axes=(None, 0, 0))(flow, x, condition).mean()


loss_fn = functools.partial(batch_loss_fn, static=static)

# xs = jnp.repeat(phis, 2, axis=1)
xs = jnp.repeat(phis, 1, axis=1)
condition = shots[10, 0:100, :]
x = xs[10]
# x = jnp.array([0.1, 0.2])
log_prob(flow, x, condition)

# %%
lr = 1e-4
optimizer = optax.chain(optax.adam(lr))
opt_state = optimizer.init(params)


@jax.jit
def step(opt_state, params, x, condition):
    loss, grad = jax.value_and_grad(loss_fn)(params, x=x, condition=condition)
    updates, opt_state = optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, grad, loss


# batch_loss_fn(params, static, xs[0:10], shots[0:10, 0:100, :])
# %%
for i in range(1, 150):
    # params, opt_state, grad, loss = step(opt_state, params, xs[0:i], shots[0:i, 0:1024, :])
    # params, opt_state, grad, loss = step(opt_state, params, xs[0:10], shots[0:10, 0:100, :])
    params, opt_state, grad, loss = step(opt_state, params, xs[i:i+1], shots[i:i+1, 0:100, :])
    print(loss)

# %%

flow = paramax.unwrap(eqx.combine(params, static))
eqx.tree_pprint(flow, short_arrays=False)
# eqx.tree_pprint(grad, short_arrays=False)
flow.bijection.loc_and_scale(condition=shots[50, 0:10, :])
# %%
fig, axs = uplt.subplots(ncols=1, nrows=2, figsize=(10, 5))

samples = flow.sample(
    key=jr.key(1234), sample_shape=(10000,), condition=shots[0, 0:1, :]
)
# axs[0].hist2d(x=samples[:, 0], y=samples[:, 1], bins=100)
# axs[0].set_aspect("equal")
axs[0].hist(samples, bins=100)

samples = flow.sample(
    key=jr.key(1234), sample_shape=(10000,), condition=shots[100, 0:10, :]
)
# axs[1].hist2d(x=samples[:, 0], y=samples[:, 1], bins=100)
# axs[1].set_aspect("equal")
axs[1].hist(samples, bins=100)

# %%
key = jr.PRNGKey(1234)

n_params = 1
n_wires = 4
n_shots = 12
n_batch = 13

x = jr.uniform(
    key=key,
    shape=(
        4,
        n_wires,
    ),
)
# x = jr.uniform(key=key, shape=(n_batch, n_shots, n_wires,))
y = jr.uniform(key=key, shape=(n_batch, n_params))

# %%
f = MLP(
    key=key, in_size=n_wires, out_size=3, width_size=4, depth=3, activation=jax.nn.relu
)
g = MLP(key=key, in_size=1, out_size=1, width_size=4, depth=3, activation=jax.nn.relu)
# ppp, static = eqx.partition(g, eqx.is_array)
# eqx.tree_pprint(static)

from rich.pretty import pprint

pprint(eqx.partition(g, eqx.is_array))

params, static = eqx.partition(g, eqx.is_array)


def loss(params, static, x, y):
    f = eqx.combine(params, static)
    yhat = f(x)
    return jnp.mean((y - yhat) ** 2)


jax.grad(functools.partial(loss, static=static), argnums=0)(
    params, x=jnp.array([0.2]), y=jnp.array([0.1])
)
# loss(params, static, jnp.array([0.2]), jnp.array([0.1]))
# %%
pinn = PermutationInvariantNeuralNetwork(f=f, g=g)

# %%
x = jr.uniform(
    key=key,
    shape=(
        n_batch,
        n_shots,
        n_wires,
    ),
)
y = jr.uniform(key=key, shape=(n_batch, n_params))

latent = jax.vmap(pinn)(x)
print(latent.shape)

# %%
print(shots.shape)
latent = jax.vmap(pinn)(shots)
print(latent.shape)

# %%
print(data)
treescope.render_array(shots[:128, :46, :], rows=(1,), columns=(2, 0))


# %%
# if __name__ == "__main__":
#     # %%
# print("Hello world")
