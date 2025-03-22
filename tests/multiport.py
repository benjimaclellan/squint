import functools
import itertools

import einops
import jax
import jax.numpy as jnp
from opt_einsum import get_symbol

from squint.ops.base import (
    create,
    destroy,
    eye,
)

# %%
dim = 2
wires = (
    0,
    1,
)
rs = jnp.ones(shape=[len(wires) * (len(wires) - 1) // 2], dtype=jnp.float64) * 0.1

# _r = jnp.einsum("a b, c d -> a b c d", create(dim), destroy(dim))
_r = jnp.einsum("a c, b d -> a c b d", create(dim), destroy(dim))
# _l = jnp.einsum("a b, c d -> a b c d", destroy(dim), create(dim))
_l = jnp.einsum("a c, b d -> a c b d", destroy(dim), create(dim))
combs = list(itertools.combinations(range(len(wires)), 2))

# %%
_subscripts_l = "a c b d"
_subscripts = [
    f"{
        ' '.join(
            [
                get_symbol(k)
                # for k in (i, i + len(wires), j, j + len(wires))
                for k in (2 * i, 2 * j, 2 * i + 1, 2 * j + 1)
                # if k in (2 * i, 2 * i + 1, 2 * j, 2 * j + 1)
            ]
        )
    }"
    " -> "
    f"{
        ' '.join(
            [get_symbol(2 * k) if k in (i, j) else '1' for k in range(len(wires))]
            + [get_symbol(2 * k + 1) if k in (i, j) else '1' for k in range(len(wires))]
        )
    }"
    # f"{
    #     ' '.join(
    #         [
    #             get_symbol(k) if k in (i, i + len(wires), j, j + len(wires)) else '1'
    #             for k in range(2 * len(wires))
    #         ]
    #     )
    # }"
    for i, j in combs
]
print(_subscripts)
# %%
_s_matrix = (
    f"({' '.join([get_symbol(2 * k) for k in range(len(wires))])}) "
    f"({' '.join([get_symbol(2 * k + 1) for k in range(len(wires))])})"
)

# _s_matrix = (
#     f"({' '.join([get_symbol(k) for k in range(len(wires))])}) "
#     f"({' '.join([get_symbol(k) for k in range(len(wires), 2 * len(wires))])})"
# )

# _s_tensor =  f"{' '.join([get_symbol(k) for k in range(2 * len(wires))])}"
_s_tensor = f"{' '.join([get_symbol(2 * k) for k in range(len(wires))])} {' '.join([get_symbol(2 * k + 1) for k in range(len(wires))])}"
# _s_tensor =  f"{' '.join([get_symbol(k) for k in range(2 * len(wires))])}"
dims = {get_symbol(k): dim for k in range(2 * len(wires))}

_h = sum(
    [r * einops.rearrange(_r, subscript) for r, subscript in zip(rs, _subscripts, strict=False)]
    + [
        r.conj() * einops.rearrange(_l, subscript)
        for r, subscript in zip(rs, _subscripts, strict=False)
    ]
)  # .reshape([dim**len(wires), dim**len(wires)]

_h_mat = einops.rearrange(_h, f"{_s_tensor} -> {_s_matrix}", **dims)
print(_h_mat)
jnp.kron(r * create(dim), destroy(dim)) + jnp.kron(destroy(dim), r * create(dim))
bs_l = jnp.kron(create(dim), destroy(dim))
bs_r = jnp.kron(destroy(dim), create(dim))
h_bs = (r * (bs_l + bs_r)).reshape(4 * (dim,))
einops.rearrange(h_bs, f"{_s_matrix} -> {_s_tensor}", **dims)

# _h = einops.rearrange(
#     _h,
#     (f"({' '.join([get_symbol(2*k) for k in range(len(wires))])}) "
#     f"({' '.join([get_symbol(2*k+1) for k in range(len(wires))])})"
#     " -> "
#     f"{' '.join([get_symbol(k) for k in range(2 * len(wires))])}"),
#     **{get_symbol(k): dim for k in range(2 * len(wires))}
# )
# sns.heatmap(jnp.abs(u))
# plt.show()
# sns.heatmap(jnp.angle(u))

# %%
u = jax.scipy.linalg.expm(
    1j * einops.rearrange(_h, f"{_s_tensor} -> {_s_matrix}", **dims)
)
print(u.conj().T @ u)
sns.heatmap(jnp.abs(u))
# %%
u = einops.rearrange(
    jax.scipy.linalg.expm(
        1j * einops.rearrange(_h, f"{_s_tensor} -> {_s_matrix}", **dims)
    ),
    f"{_s_matrix} -> {_s_tensor}",
    **dims,
)

# %%
state = jnp.zeros(shape=len(wires) * (dim,))
state = state.at[1, 0].set(1)
state = jnp.einsum(
    # (
    #  f"{' '.join([get_symbol(2*k) for k in range(len(wires))])} "
    #  f"{' '.join([get_symbol(2*k+1) for k in range(len(wires))])}"
    #  " , "
    #  f"{' '.join([get_symbol(2 * k) for k in range(len(wires))])}"
    #  " -> "
    #  f"{' '.join([get_symbol(2 * k + 1) for k in range(len(wires))])}"
    # )
    "a c b d, a c -> b d",
    u,
    state,
)
prob = jnp.abs(state) ** 2
print(state)
print(jnp.sum(prob))

from squint.utils import print_nonzero_entries

print_nonzero_entries(prob)

# %%

r = 0.1
bs_l = jnp.kron(create(dim), destroy(dim))
bs_r = jnp.kron(destroy(dim), create(dim))
h = r * (bs_l + bs_r)
u = jax.scipy.linalg.expm(1j * h)
u = u.reshape(4 * (dim,))
u = einops.rearrange(u, "a b c d -> a c b d")

state = jnp.zeros(shape=2 * (dim,))
state = state.at[1, 0].set(1)
state = jnp.einsum("a c b d , a b  -> c d", u, state)
prob = jnp.abs(state) ** 2
print(prob)
print_nonzero_entries(prob)

# %%
dim = 2
wires = (0, 1, 2)
rs = jnp.ones(shape=[len(wires) * (len(wires) - 1) // 2], dtype=jnp.float64) * 0.5
combs = list(itertools.combinations(range(len(wires)), 2))


d = {i: "yes", 2: "no"}
_h = sum(
    [
        functools.reduce(
            jnp.kron,
            [
                {i: r * create(dim), j: destroy(dim)}.get(k, eye(dim))
                for k in range(len(wires))
            ],
        )
        for r, (i, j) in zip(rs, combs, strict=False)
    ]
    + [
        functools.reduce(
            jnp.kron,
            [
                {j: r.conj() * create(dim), i: destroy(dim)}.get(k, eye(dim))
                for k in range(len(wires))
            ],
        )
        for r, (i, j) in zip(rs, combs, strict=False)
    ]
)

# %%
i, j = 1, 2
[
    [
        {j: jnp.conjugate(r) * create(dim), i: destroy(dim)}.get(k, eye(dim))
        for k in range(len(wires))
    ]
    for r, (i, j) in zip(rs, combs, strict=False)
]

# %%
_s_matrix = (
    f"({' '.join([get_symbol(2 * k) for k in range(len(wires))])}) "
    f"({' '.join([get_symbol(2 * k + 1) for k in range(len(wires))])})"
)
_s_tensor = f"{' '.join([get_symbol(2 * k) for k in range(len(wires))])} {' '.join([get_symbol(2 * k + 1) for k in range(len(wires))])}"
dims = {get_symbol(k): dim for k in range(2 * len(wires))}

u = jax.scipy.linalg.expm(1j * _h)
print(u.conj().T @ u)

u = einops.rearrange(u, f"{_s_matrix} -> {_s_tensor}", **dims)

# %%

state = jnp.zeros(shape=len(wires) * (dim,))
state = state.at[1, 0, 0].set(1)
state = jnp.einsum(
    # (
    #  f"{' '.join([get_symbol(2*k) for k in range(len(wires))])} "
    #  f"{' '.join([get_symbol(2*k+1) for k in range(len(wires))])}"
    #  " , "
    #  f"{' '.join([get_symbol(2 * k) for k in range(len(wires))])}"
    #  " -> "
    #  f"{' '.join([get_symbol(2 * k + 1) for k in range(len(wires))])}"
    # )
    "a c e b d f, a c e -> b d f",
    u,
    state,
)
prob = jnp.abs(state) ** 2
print(state)
print(jnp.sum(prob))

from squint.utils import print_nonzero_entries

print_nonzero_entries(prob)

# %%
# print(jnp.einsum(' abcdef, acd -> bdf', u, state))

# einops.rearrange(u.)

# sns.heatmap(jnp.angle(u))

# _subscripts_l = ",".join([
#     ''.join([get_symbol(2*i), get_symbol(2*i+1), get_symbol(2*j), get_symbol(2*j+1)])
#     for i, j in combs
# ])
# _subscripts_r = "".join([get_symbol(i) for i in range(2 * len(wires))])
# _subscripts = f"{_subscripts_l} -> {_subscripts_r}"
# for i, j in combs:
#     print(i, j)
#     print(get_symbol(2*i), get_symbol(2*j))
#     _subscripts = ''.join([get_symbol(2*i), get_symbol(2*i+1), get_symbol(2*j), get_symbol(2*j+1)])
#     print(_subscripts)

# jnp.einsum(_subscripts, *_ops).reshape([dim ** len(wires), dim ** len(wires)])
