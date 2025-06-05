# #%%
# import itertools

# import equinox as eqx
# import jax
# import jax.numpy as jnp
# import matplotlib.pyplot as plt
# import seaborn as sns
# from rich.pretty import pprint

# from squint.circuit import Circuit, compile
# from squint.ops.base import SharedGate
# from squint.ops.dv import Conditional, DiscreteVariableState, HGate, RZGate, XGate
# import timeit

# def benchmark(n):
#     # n = 3  # number of qubits
#     circuit = Circuit(backend="pure")
#     for i in range(n):
#         circuit.add(DiscreteVariableState(wires=(i,), n=(0,)))

#     circuit.add(HGate(wires=(0,)))
#     for i in range(n - 1):
#         circuit.add(Conditional(gate=XGate, wires=(i, i + 1)))

#     circuit.add(
#         SharedGate(op=RZGate(wires=(0,), phi=0.0 * jnp.pi), wires=tuple(range(1, n))),
#         "phase",
#     )

#     for i in range(n):
#         circuit.add(HGate(wires=(i,)))

#     # pprint(circuit)

#     params, static = eqx.partition(circuit, eqx.is_inexact_array)
#     sim = compile(static, 2, params, optimize="greedy") #.jit()

#     # Benchmark sim.probability.forward
#     def _benchmark():
#         sim.probabilities.grad(params)
#         sim.probabilities.cfim(params)
#         return

#     execution_time = timeit.repeat(_benchmark, number=3, repeat=3)
#     # print(execution_time)
#     print(f"n={n}, Best execution time: {min(execution_time):.6f} seconds")


# for n in (14,):
#     benchmark(n)
# # %%
