# Copyright 2024-2026 Benjamin MacLellan

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%
from __future__ import annotations

import functools
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Sequence, Union
import warnings

from beartype.door import is_bearable
from beartype.typing import Sequence

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import paramax
from beartype import beartype
from beartype.typing import Type
from jaxtyping import Array, PyTree
from opt_einsum.parser import get_symbol
from ordered_set import OrderedSet

from squint.interface.base import (
    Circuit,
    AbstractErasureChannel,
    AbstractGate,
    AbstractKrausChannel,
    AbstractMeasurement,
    AbstractMixedState,
    AbstractPureState,
    Block, 
    wire_sort_key
)

from squint.backends.tensornetwork.compiler import (
    circuit_to_optimized_tensor_network_contraction_path,
    circuit_to_tensors,
    PureBackend, MixedBackend,
    circuit_to_allowed_backends,
    circuit_to_wire_order,
)
from squint.math.information_matrices import qfim, cfim
dtype_complex = jnp.complex128  # TODO: make configurable

#%%
def _default_callable(*args, **kwargs):
    raise NotImplementedError("The derived callable is not implemented.")

@dataclass
class Simulator:
    backend: type[AbstractBackend] 
    subscripts: str
    path: list[tuple[int, int]]
    
    forward: Callable = _default_callable
    grad: Callable = _default_callable
    fisher_info: Callable = _default_callable
    
    @beartype
    def __init__(
        self, 
        static: PyTree,
        params: Union[PyTree, Sequence[PyTree]],
        backend: Optional[type[AbstractBackend]] = None,
        **kwargs
    ):
        holomorphic = False
        
        params = tuple(params) if isinstance(params, (list, tuple)) else (params,)
        
        model = paramax.unwrap(functools.reduce(eqx.combine, (static,) + params))
        
        backend_default = circuit_to_allowed_backends(model)
        if backend is None:
            backend = backend_default
            
        if not backend != backend_default:
            if backend == PureBackend and backend_default == MixedBackend:
                warnings.warn(f"{backend} not possible with the provided circuit, defaulting to {backend_default}.")
                backend = backend_default
                
        subscripts, path = circuit_to_optimized_tensor_network_contraction_path(model, backend=backend)

        def forward(*params):
            _circuit = paramax.unwrap(functools.reduce(eqx.combine, (static,) + params))
            # _circuit = eqx.combine(params, static)  # static in closure
            
            tensors = circuit_to_tensors(_circuit, backend=backend)
            
            return jnp.einsum(
                subscripts,
                *tensors,
                optimize=path,
            )
            
            # potentially necessary for QFIM calculations
            # *jtu.tree_map(
            #   lambda x: x.astype(dtype_complex),
            #   backend.evaluate(circuit),
            # )
            
        self.backend = backend
        self.forward = forward
        self.subscripts = subscripts
        self.path = path
        
        self.grad = jax.jacfwd(
            forward, holomorphic=holomorphic
        )
        
        return

    def jit(self, device: jax.Device = None):
        self.forward = jax.jit(self.forward, device=device)
        self.grad = jax.jit(self.grad, device=device)
        return self
        
#%%
if __name__ == "__main__":
#%%
    params, static = partition_op(circuit, "phase")
    simulator = Simulator(static=static, params=params, )

    simulator.forward(params)
    simulator.grad(params)

    simulator.jit()
    #%%
    print(simulator.forward(params))
    print(simulator.grad(params).ops['phase'].phi)


# TODO: tidy up the old Simulator class
# #%%

#     @dataclass
#     class Simulator:
#         """
#         Simulator for quantum circuits, providing callable methods for computing
#         forward, backward, and Fisher Information matrix calculations on the
#         quantum amplitudes and classical probabilities, given a set of parameters PyTrees

#         Attributes:
#             amplitudes (SimulatorQuantumAmplitudes): Object for quantum amplitudes computations.
#             probabilities (SimulatorClassicalProbabilities): Object for classical probabilities computations.
#             path (Any): Path to the simulator, can be used for saving/loading.
#             info (str, optional): Additional information about the simulator.
#         """

#         circuit: Circuit
#         backend: AbstractBackend

#         amplitudes: SimulatorQuantumAmplitudes
#         probabilities: SimulatorClassicalProbabilities

#         path: Any
#         info: str = None

#         @beartype
#         @classmethod
#         def compile(
#             cls,
#             static: PyTree,
#             *params,
#             **kwargs,
#         ):
#             """
#             Compiles the circuit into a tensor contraction function.

#             Args:
#                 static (PyTree): The static PyTree, following the `equinox` convention. These are parameters that are fixed.
#                 # dim (int): The dimension of the local Hilbert space (the same dimension across all wires).
#                 params (Sequence[PyTree]): The parameterized PyTree, following the `equinox` convention. These are parameters that will be used in gradient and Fisher information calculations.

#             Returns:
#                 sim (Simulator): A class which contains methods for computing the parameterized forward, grad, and Fisher information functions.
#             """

#             circuit = paramax.unwrap(functools.reduce(eqx.combine, (static,) + params))
#             backend = _select_backend(circuit)

#             def _tensor_func(
#                 circuit,
#                 subscripts: str,
#                 path: tuple,
#                 backend: AbstractBackend,
#             ):
#                 return jnp.einsum(
#                     subscripts,
#                     *jtu.tree_map(
#                         lambda x: x.astype(dtype_complex),
#                         backend.evaluate(circuit),
#                     ),
#                     optimize=path,
#                 )

#             optimize = kwargs.get("optimize", "greedy")
#             argnum = kwargs.get("argnum", 0)

#             dtype_complex = jnp.complex128  # TODO: Add to config

#             subscripts = backend.subscripts(circuit)
#             path, info = _path(circuit, backend, optimize=optimize)

#             wires = circuit.wires

#             wires_ptrace = OrderedSet(
#                 sorted(
#                     dict.fromkeys(
#                         itertools.chain.from_iterable(
#                             op.wires
#                             for op in circuit.unwrap()
#                             if isinstance(op, AbstractErasureChannel)
#                         )
#                     ),
#                     key=wire_sort_key,
#                 )
#             )

#             # wires_ptrace = OrderedSet(
#             #     sum(
#             #         (
#             #             op.wires
#             #             for op in circuit.unwrap()
#             #             if isinstance(op, AbstractErasureChannel)
#             #         ),
#             #         (),
#             #     )
#             # )

#             _tensor = functools.partial(
#                 _tensor_func,
#                 subscripts=subscripts,
#                 path=path,
#                 backend=backend,
#             )

#             def _forward_state_func(static: PyTree, *params):
#                 circuit = paramax.unwrap(functools.reduce(eqx.combine, (static,) + params))
#                 return _tensor(circuit)

#             _forward_state = functools.partial(_forward_state_func, static)

#             if backend is PureBackend:

#                 def _forward_prob(*params: Sequence[PyTree]):
#                     return jnp.abs(_forward_state(*params)) ** 2

#             elif backend is MixedBackend:

#                 def _forward_prob(*params: Sequence[PyTree]):
#                     # remove wires that have been traced out
#                     _subscripts_tmp = [
#                         get_symbol(i) for i in range(len(wires - wires_ptrace))
#                     ]
#                     _subscripts = (
#                         "".join(_subscripts_tmp + _subscripts_tmp)
#                         + "->"
#                         + "".join(_subscripts_tmp)
#                     )
#                     return jnp.abs(jnp.einsum(_subscripts, _forward_state(*params)))
#             else:
#                 raise RuntimeError("Backend not found or provided.")

#             _grad_state_holomorphic = jax.jacfwd(
#                 _forward_state, argnums=argnum, holomorphic=True
#             )
#             _grad_prob = jax.jacfwd(_forward_prob, argnums=argnum)

#             # _grad_state_holomorphic = jax.jacrev(
#             #     _forward_state, argnums=argnum, holomorphic=True
#             # )
#             # _grad_prob = jax.jacrev(_forward_prob, argnums=argnum)

#             def _grad_state(*params: Sequence[PyTree]):
#                 params = jtu.tree_map(lambda x: x.astype(dtype_complex), params)
#                 return _grad_state_holomorphic(*params)

#             if backend is PureBackend:
#                 _qfim_state = functools.partial(
#                     quantum_fisher_information_matrix, _forward_state, _grad_state
#                 )

#             elif backend is MixedBackend:

#                 def _qfim_state(*params):
#                     raise NotImplementedError("QFIM for mixed states not implemented")

#             else:
#                 raise RuntimeError("Backend not found or provided.")

#             _cfim_state = functools.partial(
#                 classical_fisher_information_matrix, _forward_prob, _grad_prob
#             )

#             return cls(
#                 circuit=circuit,
#                 backend=backend,
#                 amplitudes=SimulatorQuantumAmplitudes(
#                     forward=_forward_state,
#                     grad=_grad_state,
#                     qfim=_qfim_state,
#                 ),
#                 probabilities=SimulatorClassicalProbabilities(
#                     forward=_forward_prob,
#                     grad=_grad_prob,
#                     cfim=_cfim_state,
#                 ),
#                 path=path,
#                 info=info,
#             )

#         @property
#         def subscripts(self):
#             return self.backend.subscripts(self.circuit)

#         @property
#         def wires(self):
#             if self.backend is PureBackend:
#                 return self.circuit.wires
#             elif self.backend is MixedBackend:
#                 return self.circuit.wires + self.circuit.wires

#         def display_wires(self):
#             return ",".join([f"{wire.idx}" for wire in self.wires])

#         def jit(self, device: jax.Device = None):
#             """
#             JIT (just-in-time) compile the simulator methods.
#             Args:
#                 device (jax.Device, optional): Device to compile the methods on. Defaults to None, which uses the first available device.
#             """
#             if not device:
#                 device = jax.devices()[0]

#             return Simulator(
#                 circuit=self.circuit,
#                 backend=self.backend,
#                 amplitudes=self.amplitudes.jit(device=device),
#                 probabilities=self.probabilities.jit(device=device),
#                 path=self.path,
#                 info=self.info,
#             )

#         def sample(self, key: jr.PRNGKey, params: PyTree, shape: tuple[int, ...]):
#             """
#             Sample from the quantum circuit using the provided parameters and a random key.
#             Args:
#                 key (jr.PRNGKey): Random key for sampling.
#                 params (PyTree): Parameters for the quantum circuit, partitioned via `eqx.partition`.
#                 shape (tuple[int, ...]): Shape of the output samples.
#             Returns:
#                 samples (jnp.ndarray): Samples drawn from the quantum circuit.
#             """
#             pr = self.probabilities.forward(params)
#             idx = jnp.nonzero(pr)
#             samples = einops.rearrange(
#                 jr.choice(key=key, a=jnp.stack(idx), p=pr[idx], shape=shape, axis=1),
#                 "s ... -> ... s",
#             )
#             return samples


# %%
