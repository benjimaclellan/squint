# cut = 3
# op = BeamSplitter(wires=(0, 1), r=0.03, phi=0.2)
# op = Phase(wires=(0,), phi=jnp.pi/4)
# u = op(cut=cut)
# print(jnp.conjugate(u) @ u)
# print(jnp.einsum("aAbB,AcBd->abcd", jnp.conjugate(u), u))
# jnp.isclose(bs_left, bs_l)
