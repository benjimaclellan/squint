import jax
import jax.numpy as jnp
from jax import lax
import einops


def nest_vmap(in_axes_outer=2, in_axes_inner=2):
    def decorator(func):
        vmapped_func = jax.vmap(jax.vmap(func, in_axes=in_axes_inner), in_axes=in_axes_outer)
        return vmapped_func
    return decorator


# @nest_vmap(in_axes_outer=0, in_axes_inner=0)
@nest_vmap(in_axes_outer=2, in_axes_inner=2)
def perm_ryser(mat):  # pragma: no cover
    """
    Returns the permanent of a matrix using the Ryser formula in Gray ordering.

    The code is a re-implementation from a Python 2 code found in
    `Permanent code golf
    <https://codegolf.stackexchange.com/questions/97060/calculate-the-permanent-as-quickly-as-possible>`_
    using Numba.

    Args:
        mat (jnp.array) : a square array.

    Returns:
        float or complex: the permanent of matrix ``M``
    """
    n = len(mat)
    if n == 0:
        return mat.dtype.type(1.0)
    # row_comb keeps the sum of previous subsets.
    # Every iteration, it removes a term and/or adds a new term
    # to give the term to add for the next subset
    row_comb = jnp.zeros((n), dtype=mat.dtype)
    total = 0
    old_grey = 0
    sign = +1
    binary_power_dict = [2**i for i in range(n)]
    num_loops = 2**n
    for k in range(0, num_loops):
        bin_index = (k + 1) % num_loops
        reduced = jnp.prod(row_comb)
        total += sign * reduced
        new_grey = bin_index ^ (bin_index // 2)
        grey_diff = old_grey ^ new_grey
        grey_diff_index = binary_power_dict.index(grey_diff)
        new_vector = mat[grey_diff_index]
        direction = (old_grey > new_grey) - (old_grey < new_grey)
        for i in range(n):
            row_comb = row_comb.at[i].add(new_vector[i] * direction)
        sign = -sign
        old_grey = new_grey
    return total


# %%
# import numpy as np


def per(mtx, column, selected, prod, output=False):
    """
    Row expansion for the permanent of matrix mtx.
    The counter column is the current column,
    selected is a list of indices of selected rows,
    and prod accumulates the current product.
    """
    if column == mtx.shape[1]:
        if output:
            print(selected, prod)
        return prod
    else:
        result = 0
        for row in range(mtx.shape[0]):
            if not row in selected:
                result = result + per(
                    mtx, column + 1, selected + [row], prod * mtx[row, column]
                )
        return result


def permanent(mat):
    """
    Returns the permanent of the matrix mat.
    """
    return per(mat, 0, [], 1)


@nest_vmap(in_axes_outer=2, in_axes_inner=2)
def perm_ryser_scan(mat):
    n = mat.shape[0]
    binary_power_dict = jnp.array([2**i for i in range(n)], dtype=jnp.int32)
    num_loops = 2**n

    # Inner scan function to update row_comb
    def inner_scan(carry, i):
        row_comb, old_grey, sign = carry
        new_grey = i ^ (i // 2)
        grey_diff = old_grey ^ new_grey
        grey_diff_index = jnp.argmax(binary_power_dict == grey_diff)
        new_vector = mat[grey_diff_index]
        direction = jnp.int32(old_grey > new_grey) - jnp.int32(old_grey < new_grey)
        row_comb = row_comb + new_vector * direction
        return (row_comb, new_grey, sign), None

    # Outer scan function to accumulate total
    def outer_scan(carry, k):
        total, row_comb, old_grey, sign = carry
        (row_comb, old_grey, sign), _ = lax.scan(inner_scan, (row_comb, old_grey, sign), jnp.array([k]))
        reduced = jnp.prod(row_comb)
        total += sign * reduced
        sign = -sign
        return (total, row_comb, old_grey, sign), None

    # Initial values
    row_comb = jnp.zeros((n), dtype=mat.dtype)
    total = jnp.array(0.0, dtype=mat.dtype)
    old_grey = jnp.array(0, dtype=jnp.int32)
    sign = jnp.array(-1, dtype=mat.dtype)

    # Perform the outer scan
    (total, _, _, _), _ = lax.scan(outer_scan, (total, row_comb, old_grey, sign), jnp.arange(1, num_loops, dtype=jnp.int32))

    return total


def get_fixed_sum_tuples(length, total):
    """Generate all tuples of a given length that sum to a specified total."""
    if length == 1:
        yield (total,)
        return

    for i in range(total + 1):
        for t in get_fixed_sum_tuples(length - 1, total - i):
            yield (i,) + t
            

def compile_Aij_indices(i_s: jnp.array, j_s: jnp.array, m: int, n: int):
    """Compile all indices for generating the $A_{ij}$ matrices for all i and j combinations."""
    # checkify.check(
    #     jnp.all(i_s.sum(axis=1) == n), f"Some input bases do not have n={n} photons."
    # )
    # checkify.check(
    #     jnp.all(j_s.sum(axis=1) == n), f"Some output bases do not have n={n} photons."
    # )

    unitary_inds = jnp.indices((m, m))

    def repeated_indices(i_basis: jnp.array, j_basis: jnp.array):
        rectangular = jnp.concat(
            [
                einops.repeat(
                    unitary_inds[:, :, i : i + 1],
                    "ind row col -> ind row (rep col)",
                    rep=i_basis[i],
                )
                for i in range(m)
            ],
            axis=2,
        )

        square = jnp.concat(
            [
                einops.repeat(
                    rectangular[:, i : i + 1, :],
                    "ind row col -> ind (rep row) col",
                    rep=j_basis[i],
                )
                for i in range(m)
            ],
            axis=1,
        )

        return square

    transition_inds = jnp.array(
        [[repeated_indices(i_basis, j_basis) for j_basis in j_s] for i_basis in i_s]
    )
    return transition_inds


@jax.jit
def compute_transition_amplitudes(unitary: jnp.array, transition_inds: jnp.array):
    """Calculates all i -> j transition amplitudes in a jit-able manner."""
    a_ijs = unitary[transition_inds[:, :, 0, :, :], transition_inds[:, :, 1, :, :]]

    # swapping axes required when using recursive permanent function
    a_ijs_swapaxes = einops.rearrange(a_ijs, "i o a b -> a b i o")
    transition_amplitudes = permanent(a_ijs_swapaxes)  # fastest after jit of the three

    # if using Ryser algorithm, swapping axes not needed as vmap over can help (if in_axes are 0, 0), otherwise 2, 2
    # transition_amplitudes = perm_ryser(a_ijs_swapaxes)
    # transition_amplitudes = perm_ryser_scan(a_ijs_swapaxes)
    return transition_amplitudes