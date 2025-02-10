# %%
import equinox as eqx
import jax
import jax.numpy as jnp

from squint.circuit import Circuit

# %%


# def classical_fisher_information(_params):
#     _grad = sim.grad(_params)
#     # cfim = jnp.sum(grad.ops['phase'].phi ** 2 / (pr + 1e-14))
#     # cfim = jnp.nansum(grad.ops['phase'].phi ** 2 / (pr + 1e-14))
#     A = _grad.ops["phase"].phi ** 2
#     B = pr
#     cfim = jnp.sum(A * (B / (B**2 + 1e-18)))  # soft-approximation of CFI
#     return cfim


# def loss(params):
#     grad = sim.grad(params)
#     A = grad.ops["phase"].phi ** 2
#     val = jnp.sum(A)  # soft-approximation of CFI
#     return val


def print_nonzero_entries(arr):
    nonzero_indices = jnp.array(jnp.nonzero(arr)).T
    nonzero_values = arr[tuple(nonzero_indices.T)]
    for idx, value in zip(nonzero_indices, nonzero_values):
        print(f"Index: {jnp.array(idx)}, Value: {value}")


def partition_op(circuit, name):  # todo: allow multiple names
    def select(circuit: Circuit, name: str):
        """Sets all leaves to `True` for a given op key from the given Pytree)"""
        get_leaf = lambda t: t.ops[name]
        null = jax.tree_map(lambda _: True, circuit.ops[name])
        return eqx.tree_at(get_leaf, circuit, null)

    def mask(val: str, mask1, mask2):
        """Logical AND mask over Pytree"""
        if isinstance(mask1, bool) and isinstance(mask2, bool):
            if mask1 and mask2:
                return True
        else:
            return False

    _params = eqx.filter(circuit, eqx.is_inexact_array, inverse=True, replace=True)
    _op = select(circuit, name)

    filter = jax.tree_map(mask, circuit, _params, _op)

    params, static = eqx.partition(circuit, filter_spec=filter)

    return params, static


# %%
