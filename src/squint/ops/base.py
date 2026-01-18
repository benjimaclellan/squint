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

# %%
import functools
import itertools
from collections import OrderedDict
from typing import Optional, Union
from uuid import uuid4

import equinox as eqx
import jax.numpy as jnp
import scipy as sp
from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Callable, Sequence

from squint.ops.gellmann import gellmann


_wire_id = itertools.count(1)


class AbstractDoF(eqx.Module):
    """
    Abstract base class for degrees of freedom (DoF) in quantum systems.

    Degrees of freedom classify the physical encoding of quantum information.
    Each DoF subclass represents a different physical implementation for
    encoding quantum states, which can be useful for enforcing circuit
    validity and distinguishing between different physical platforms.

    Subclasses:
        DV: Discrete variable systems (qubits, qudits)
        CV: Continuous variable systems (optical modes)
        TimeBin: Time-bin encoded photonic systems
        FreqBin: Frequency-bin encoded photonic systems
        Spatial: Spatial mode encoding
    """

    pass


class DV(AbstractDoF):
    """
    Discrete variable degree of freedom.

    Represents finite-dimensional quantum systems such as qubits (dim=2)
    or qudits (dim>2). These systems have a finite number of basis states
    and are commonly implemented in platforms like superconducting circuits,
    trapped ions, and spin systems.

    Example:
        ```python
        wire = Wire(dim=2, dof=DV, idx=0)  # A qubit wire
        ```
    """

    pass


class CV(AbstractDoF):
    """
    Continuous variable degree of freedom.

    Represents infinite-dimensional Fock space systems, typically optical
    modes with photon number states. In practice, the Hilbert space is
    truncated at a finite photon number cutoff specified by the wire dimension.

    Example:
        ```python
        wire = Wire(dim=10, dof=CV, idx=0)  # Optical mode with 10 photon cutoff
        ```
    """

    pass


class TimeBin(AbstractDoF):
    """
    Time-bin encoded degree of freedom.

    Represents photonic qubits/qudits encoded in discrete time bins.
    Information is encoded in the arrival time of single photons,
    commonly used in fiber-based quantum communication.

    Example:
        ```python
        wire = Wire(dim=2, dof=TimeBin, idx=0)  # Time-bin qubit
        ```
    """

    pass


class FreqBin(AbstractDoF):
    """
    Frequency-bin encoded degree of freedom.

    Represents photonic qubits/qudits encoded in discrete frequency modes.
    Information is encoded in the spectral properties of photons,
    useful for wavelength-division multiplexing in quantum networks.

    Example:
        ```python
        wire = Wire(dim=4, dof=FreqBin, idx=0)  # 4-level frequency-bin qudit
        ```
    """

    pass


class Spatial(AbstractDoF):
    """
    Spatial mode encoded degree of freedom.

    Represents quantum information encoded in spatial modes of light,
    such as different paths in an interferometer or transverse spatial
    modes (e.g., orbital angular momentum modes).

    Example:
        ```python
        wire = Wire(dim=2, dof=Spatial, idx=0)  # Dual-rail spatial encoding
        ```
    """

    pass


class Wire(eqx.Module):
    """
    Represents a quantum subsystem (wire) in a circuit.

    A Wire defines a single quantum subsystem with a specific Hilbert space dimension
    and degree of freedom type. Wires are the fundamental building blocks that connect
    quantum operations in a circuit, determining how operators act on different parts
    of the composite quantum system.

    Attributes:
        dim (int): The dimension of the local Hilbert space. For qubits, dim=2;
            for qudits, dim>2; for Fock spaces, dim is the photon number cutoff.
        dof (type[AbstractDoF]): The type of degree of freedom this wire represents
            (e.g., DV for discrete variable, CV for continuous variable).
        idx (str | int): Unique identifier for the wire, used to track which
            operations act on which subsystems.

    Example:
        ```python
        from squint.ops.base import Wire, DV, CV

        # Create a qubit wire
        qubit = Wire(dim=2, dof=DV, idx=0)

        # Create an optical mode wire with photon cutoff of 5
        mode = Wire(dim=5, dof=CV, idx="signal")

        # Use wires in operations
        from squint.ops.dv import DiscreteVariableState, HGate
        state = DiscreteVariableState(wires=(qubit,), n=(0,))
        gate = HGate(wires=(qubit,))
        ```
    """

    idx: int | str = 0
    dim: int
    dof: type[AbstractDoF]

    @beartype
    def __init__(
        self,
        dim: int,
        dof: Optional[type[AbstractDoF]] = AbstractDoF,
        idx: Optional[str | int] = None,
    ):
        """
        Initialize a Wire.

        Args:
            dim (int): The dimension of the local Hilbert space. Must be >= 2.
                For qubits use dim=2, for qudits use dim>2, for Fock spaces
                this is the photon number cutoff.
            dof (type[AbstractDoF], optional): Type of degree of freedom that
                this wire represents. Can be used to enforce circuit validity
                and distinguish between different physical encodings.
                Defaults to AbstractDoF.
            idx (str | int, optional): Unique identifier for the wire. If not
                provided, a random id is generated.

        Raises:
            ValueError: If dim < 2.
        """
        if dim < 2:
            raise ValueError("Dimension should be 2 or greater.")
        self.dim = dim
        self.dof = dof
        # self.idx = idx if idx is not None else str(uuid4())
        self.idx = idx if idx is not None else f"__w{next(_wire_id)}"


@functools.cache
def create(dim):
    """
    Returns the create operator for a Hilbert space of dimension `dim`.
    The create operator is a matrix that adds one excitation to a quantum system,
    effectively increasing the energy level by one.
    Args:
        dim (int): The dimension of the Hilbert space.
    Returns:
        jnp.ndarray: A 2D array of shape (dim, dim) representing the create operator.
    """
    return jnp.diag(jnp.sqrt(jnp.arange(1, dim)), k=-1)


@functools.cache
def destroy(dim):
    """
    Returns the destroy operator for a Hilbert space of dimension `dim`.
    The destroy operator is a matrix that annihilates one excitation of a quantum system,
    effectively reducing the energy level by one.
    Args:
        dim (int): The dimension of the Hilbert space.
    Returns:
        jnp.ndarray: A 2D array of shape (dim, dim) representing the destroy operator.
    """
    return jnp.diag(jnp.sqrt(jnp.arange(1, dim)), k=1)


@functools.cache
def eye(dim):
    """
    Returns the identity operator for a Hilbert space of dimension `dim`.

    Args:
        dim (int): The dimension of the Hilbert space.
    Returns:
        jnp.ndarray: A 2D array of shape (dim, dim) representing the identity operator.
    """
    return jnp.eye(dim)


@functools.cache
def bases(dim):
    """

    Returns the computational basis indices for a Hilbert space of dimension `dim`.

    Args:
        dim (int): The dimension of the Hilbert space.

    Returns:
        jnp.ndarray: A 1D array of shape (dim,) containing the indices of the computational bases.
    """
    return jnp.arange(dim)


@functools.cache
def dft(dim):
    """
    Returns the discrete Fourier transform matrix of dimension `dim`.

    Args:
        dim (int): The dimension of the DFT matrix.

    Returns:
        jnp.ndarray: A 2D array of shape (dim, dim) representing the DFT matrix.
    """
    return jnp.array(sp.linalg.dft(dim, scale="sqrtn"))


@functools.cache
def basis_operators(dim):
    """
    Return a basis of orthogonal Hermitian operators on a Hilbert space of
    dimension $d$, with the identity element in the last place.
    i.e., the Gell-Mann operators (for dim=2, these are the four Pauli operators).

    Args:
        dim (int): The dimension of the Hilbert space.

    Returns:
        jnp.ndarray: A 3D array of shape (n_operators, dim, dim) containing the basis operators.
    """
    return jnp.array(
        [
            gellmann(j, k, dim)
            for j, k in itertools.product(range(1, dim + 1), repeat=2)
        ],
        jnp.complex128,
    )


class AbstractOp(eqx.Module):
    """
    An abstract base class for all quantum objects, including states, gates, channels, and measurements.
    It provides a common interface for various quantum objects, ensuring consistency and reusability across different types
    of quantum operations.

    Attributes:
        wires (tuple[int, ...]): A tuple of nonnegative integers representing the quantum wires
                                 on which the operation acts. Each wire corresponds to a specific subsystem
                                 in the composite quantum system.
    """

    wires: tuple[Wire, ...]

    def __init__(
        self,
        wires: Sequence[Wire],
    ):
        """
        Initializes the AbstractOp instance.

        Args:
            wires (tuple[int, ...], optional): A tuple of nonnegative integers representing the quantum wires
                                               on which the operation acts. Defaults to (0, 1).

        Raises:
            TypeError: If any wire in the provided tuple is not a nonnegative integer.
        """
        # if not all([wire >= 0 for wire in wires]):
        # raise TypeError("All wires must be nonnegative ints.")
        self.wires = wires
        return

    def unwrap(self):
        """
        A base method for unwrapping an operator into constituent parts, important in, e.g., shared weights across operators.

        This method can be overridden by subclasses to provide additional unwrapping functionality, such as
        decomposing composite operations into their components.

        Returns:
            ops (tuple[AbstractOp]): A tuple of AbstractOp which represent the constituent ops.
        """
        return (self,)


class AbstractState(AbstractOp):
    r"""
    An abstract base class for all quantum states.
    """

    def __init__(
        self,
        wires: Sequence[Wire],
    ):
        super().__init__(wires=wires)
        return

    def __call__(self, dim: int):
        raise NotImplementedError


class AbstractPureState(AbstractState):
    r"""
    An abstract base class for all pure quantum states, equivalent to the state vector formalism.
    Pure states are associated with a Hilbert space of size
    $|\psi\rangle \in \mathcal{H}^{d_1 \times \dots \times d_w}$
    where $w$ = `len(wires)` and $d$ is assigned at compile time.
    """

    def __init__(
        self,
        wires: Sequence[Wire],
    ):
        super().__init__(wires=wires)
        return

    def __call__(self, dim: int):
        raise NotImplementedError


class AbstractMixedState(AbstractState):
    r"""
    An abstract base class for all mixed quantum states, equivalent to the density matrix formalism.
    Mixed states are associated with a Hilbert space of size
    $\rho \in \mathcal{H}^{d_1 \times \dots \times d_w \times d_1 \times \dots \times d_w}$
    where $w$ = `len(wires)` and $d$ is assigned at compile time.
    """

    def __init__(
        self,
        wires: Sequence[Wire],
    ):
        super().__init__(wires=wires)
        return

    def __call__(self, dim: int):
        raise NotImplementedError


class AbstractGate(AbstractOp):
    r"""
    An abstract base class for all unitary quantum gates, which transform an input state in a reversible way.
    $U \in \mathcal{H}^{d_1 \times \dots \times d_w \times d_1 \times \dots \times d_w}$
    where $w$ = `len(wires)` and $d$ is assigned at compile time.
    """

    def __init__(
        self,
        wires: Sequence[Wire],
    ):
        super().__init__(wires=wires)
        return

    def __call__(self, dim: int):
        raise NotImplementedError


class AbstractChannel(AbstractOp):
    r"""
    An abstract base class for quantum channels, including channels expressed as Kraus operators, erasure (partial trace), and others.
    """

    def __init__(
        self,
        wires: Sequence[Wire],
    ):
        super().__init__(wires=wires)
        return

    def unwrap(self):
        return (self,)

    def __call__(self, dim: int):
        raise NotImplementedError


class AbstractMeasurement(AbstractOp):
    r"""
    An abstract base class for quantum measurements. Currently, this is not supported, and measurements are projective measurements in the computational basis.
    """

    def __init__(
        self,
        wires: Sequence[Wire],
    ):
        super().__init__(wires=wires)
        return

    def __call__(self, dim: int):
        raise NotImplementedError


class SharedGate(AbstractGate):
    r"""
    A class representing a shared quantum gate, which allows for the sharing of parameters or attributes
    across multiple copies of a quantum operation. This is useful for scenarios where multiple gates
    share the same underlying structure or parameters, such as in variational quantum circuits.
    This is most commonly used when applying the same parameterized gate across different wires,
    e.g., phase gates, for studying phase estimation protocols.

    Attributes:
        op (AbstractOp): The base quantum operation that is shared across multiple copies.
        copies (Sequence[AbstractOp]): A sequence of copies of the base operation, each acting on different wires.
        where (Callable): A function that determines which attributes of the operation are shared across copies.
        get (Callable): A function that retrieves the shared attributes from the base operation.
    """

    op: AbstractOp
    copies: Sequence[AbstractOp]
    where: Callable
    get: Callable

    @beartype
    def __init__(
        self,
        op: AbstractOp,
        wires: Union[Sequence[Wire], Sequence[Sequence[Wire]]],
        where: Optional[Callable] = None,
        get: Optional[Callable] = None,
    ):
        # todo: handle the wires coming from both main and the shared wires
        copies = [eqx.tree_at(lambda op: op.wires, op, (wire,)) for wire in wires]
        self.copies = copies
        self.op = op

        if is_bearable(wires, Sequence[Wire]):
            wires = op.wires + wires

        elif is_bearable(wires, Sequence[Sequence[Wire]]):
            wires = op.wires + tuple(itertools.chain.from_iterable(wires))

        # wires = op.wires + wires
        super().__init__(wires=wires)

        # use a default where/get sharing mechanism, such that all ArrayLike attributes are shared exactly
        attrs = [key for key, val in op.__dict__.items() if eqx.is_array_like(val)]

        if where is None:  # todo: check  get/where functions if it is user-defined
            where = lambda pytree: sum(
                [[copy.__dict__[key] for copy in pytree.copies] for key in attrs], []
            )
        if get is None:
            get = lambda pytree: sum(
                [[pytree.op.__dict__[key] for _ in pytree.copies] for key in attrs], []
            )
        self.where = where
        self.get = get

        return

    def __check_init__(self):
        return object.__setattr__(
            self,
            "__dict__",
            eqx.tree_at(self.where, self, replace_fn=lambda _: None).__dict__,
        )

    def unwrap(self):
        """Unwraps the shared ops for compilation and contractions."""
        _self = eqx.tree_at(
            self.where, self, self.get(self), is_leaf=lambda leaf: leaf is None
        )
        return [_self.op] + [op for op in _self.copies]


class AbstractKrausChannel(AbstractChannel):
    r"""
    An abstract base class for quantum channels expressed as Kraus operators.
    The channel $K$ is of shape $(d_1 \times \dots \times d_w \times d_1 \times \dots \times d_w)$
    where $w$ = `len(wires)` and $d$ is assigned at compile time.
    """

    def __init__(
        self,
        wires: Sequence[Wire],
    ):
        super().__init__(wires=wires)
        return

    def unwrap(self):
        return (self,)

    def __call__(self, dim: int):
        raise NotImplementedError


class AbstractErasureChannel(AbstractChannel):
    """
    This channel traces out the local Hilbert space associated with the `wires`
    """

    @beartype
    def __init__(self, wires: Sequence[Wire]):
        super().__init__(wires=wires)
        return

    def __call__(self, dim: int):
        return None


class Block(eqx.Module):
    """
    A block operation that groups a sequence of quantum operations.

    Blocks allow organizing multiple operations into a single logical unit.
    They can be nested within circuits or other blocks, and support the same
    `add` and `unwrap` interface as Circuit. Unlike Circuit, Block does not
    specify a backend and is purely for organizational purposes.

    Attributes:
        ops (OrderedDict): Ordered dictionary mapping keys to operations or nested blocks.

    Example:
        ```python
        from squint.ops.base import Block, Wire
        from squint.ops.dv import RXGate, RYGate

        wire = Wire(dim=2, idx=0)
        block = Block()
        block.add(RXGate(wires=(wire,), phi=0.1), "rx")
        block.add(RYGate(wires=(wire,), phi=0.2), "ry")

        # Use in a circuit
        circuit.add(block, "rotation_block")
        ```
    """

    # TODO: remove copied functionality from `Circuit`, instead add function for converting Block to Circuit
    ops: OrderedDict[Union[str, int], Union[AbstractOp, "Block"]]
    # _backend: Literal["pure", "mixed"]

    @beartype
    def __init__(self):
        """
        Initialize an empty Block.

        Creates a new Block with no operations. Operations can be added
        using the `add` method.
        """
        self.ops = OrderedDict()
        # self._backend = backend

    @property
    def wires(self) -> set[int]:
        """
        Get all wires used by operations in this block.

        Returns:
            set[Wire]: Set of all Wire objects that operations in this block act on.
        """
        return set(sum((op.wires for op in self.unwrap()), ()))

    @beartype
    def add(self, op: Union[AbstractOp, "Block"], key: str = None) -> None:
        """
        Add an operator to the block.

        Operators are added sequentially. When this block is used in a circuit,
        the operations will be applied in the order they were added.

        Args:
            op (AbstractOp | Block): The operator or nested block to add.
            key (str, optional): A string key for indexing into the block's ops
                dictionary. If None, an integer counter is used as the key.
        """

        if key is None:
            key = len(self.ops)
        self.ops[key] = op

    def unwrap(self) -> tuple[AbstractOp]:
        """
        Unwrap all operators in the block into a flat tuple.

        Recursively calls `unwrap()` on all contained operations and nested
        blocks to produce a flat sequence of atomic operations.

        Returns:
            tuple[AbstractOp]: Flattened tuple of all operations in order.
        """
        return tuple(
            op for op_wrapped in self.ops.values() for op in op_wrapped.unwrap()
        )
