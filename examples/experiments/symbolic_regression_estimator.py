# %%
import functools
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import tqdm
from beartype import beartype


if __name__ == "__main__":
    # %%
    key = jr.PRNGKey(1234)
    model = eqx.nn.MLP(
        in_size=2,
        out_size=1,
        width_size=8,
        depth=3,
        activation=jax.nn.elu,
        dtype=jnp.float64,
        key=key,
    )

    def interferometer(phi):
        return jnp.array([jnp.cos(phi / 2), jnp.sin(phi / 2)])

    @functools.partial(jax.vmap, in_axes=(None, None, 0))
    def squared_error(params, static, phi):
        model = eqx.combine(params, static)
        measurements = interferometer(phi)
        estimate = model(measurements)
        return (estimate - phi) ** 2

    def loss_fn(params, phi):
        return squared_error(params, static, phi).mean()

    params, static = eqx.partition(model, eqx.is_array)

    phi = jnp.array([0.0, 1.0, 2.0])
    loss_fn(params, phi)

    # %%
    optimizer = optax.chain(optax.adam(1e-2))
    opt_state = optimizer.init(params)

    @jax.jit
    def update(opt_state, params, phi):
        val, grad = jax.value_and_grad(loss_fn)(params, phi)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, val

    update(opt_state, params, phi)
    # %%
    key = jr.PRNGKey(time.time_ns())
    df = []
    n_steps = 300
    pbar = tqdm.tqdm(range(n_steps), desc="Training", unit="it")
    for _step in pbar:
        key, subkey = jr.split(key)
        phi = jr.uniform(key=key, shape=(32,), minval=0.0, maxval=2.0)
        params, opt_state, val = update(opt_state, params, phi)

        pbar.set_postfix({"loss": val})
        pbar.update(1)

        df.append(val)

    # %%
    import matplotlib.pyplot as plt

    plt.plot(df)

    # %%
    phi = jr.uniform(key=key, shape=(256,), minval=0.0, maxval=2.0)
    model = eqx.combine(params, static)
    measurements = jax.vmap(interferometer)(phi)
    estimate = jax.vmap(model)(measurements)

    # %%
    fig, ax = plt.subplots()

    ax.scatter(phi, estimate)
    fig.show()

    # %%
    import pysr

    # %%
    X = measurements
    y = estimate
    # %%
    model = pysr.PySRRegressor(
        maxsize=50,
        niterations=100,  # < Increase me for better results
        binary_operators=["+", "*"],
        unary_operators=[
            "cos",
            "exp",
            "sin",
            "acos",
            "asin",
            # "inv(x) = 1/x",
            # ^ Custom operator (julia syntax)
        ],
        # extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        # ^ Custom loss function (julia syntax)
    )

    model.fit(X, y)
