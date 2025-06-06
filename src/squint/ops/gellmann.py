"""
The code for the `gellman` function is adapted from the PySME project,
which is licensed under the MIT license.

Source:
https://pysme.readthedocs.io/en/latest/_modules/gellmann.html
.. module:: gellmann.py
   :synopsis: Generate generalized Gell-Mann matrices
.. moduleauthor:: Jonathan Gross <jarthurgross@gmail.com>

Functions to generate the generalized Pauli (i.e., Gell-Mann matrices)
"""

# %%

import jax.numpy as jnp


def gellmann(j, k, d):
    r"""Returns a generalized Gell-Mann matrix of dimension d. According to the
    convention in *Bloch Vectors for Qubits* by Bertlmann and Krammer (2008),
    returns :math:`\Lambda^j` for :math:`1\leq j=k\leq d-1`,
    :math:`\Lambda^{kj}_s` for :math:`1\leq k<j\leq d`,
    :math:`\Lambda^{jk}_a` for :math:`1\leq j<k\leq d`, and
    :math:`I` for :math:`j=k=d`.

    :param j: First index for generalized Gell-Mann matrix
    :type j:  positive integer
    :param k: Second index for generalized Gell-Mann matrix
    :type k:  positive integer
    :param d: Dimension of the generalized Gell-Mann matrix
    :type d:  positive integer
    :returns: A genereralized Gell-Mann matrix.
    :rtype:   numpy.array

    """

    if j > k:
        gjkd = jnp.zeros((d, d), dtype=jnp.complex64)
        gjkd = gjkd.at[j - 1, k - 1].set(1.0)
        gjkd = gjkd.at[k - 1, j - 1].set(1.0)
    elif k > j:
        gjkd = jnp.zeros((d, d), dtype=jnp.complex64)
        gjkd = gjkd.at[j - 1, k - 1].set(-1.0j)
        gjkd = gjkd.at[k - 1, j - 1].set(1.0j)
    elif j == k and j < d:
        gjkd = jnp.sqrt(2 / (j * (j + 1))) * jnp.diag(
            jnp.array(
                [
                    1 + 0.0j if n <= j else (-j + 0.0j if n == (j + 1) else 0 + 0.0j)
                    for n in range(1, d + 1)
                ],
                dtype=jnp.complex64,
            )
        )
    else:
        gjkd = jnp.diag(
            jnp.array([1 + 0.0j for n in range(1, d + 1)], dtype=jnp.complex64)
        )

    return gjkd


# %%
