# Copyright 2024-2025 Benjamin MacLellan

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import einops
import jax
import jax.numpy as jnp


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

    return transition_amplitudes
