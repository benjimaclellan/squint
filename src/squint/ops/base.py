# %%
import functools
import itertools
from typing import Optional, Union

import equinox as eqx
import jax.numpy as jnp
import scipy as sp
from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Callable, Sequence

from squint.ops.gellmann import gellmann

WiresTypes = Union[int, str]


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
    return sp.linalg.dft(dim, scale="sqrtn")


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

    wires: tuple[WiresTypes, ...]

    def __init__(
        self,
        wires: Sequence[WiresTypes],
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
        wires: Sequence[WiresTypes],
    ):
        super().__init__(wires=wires)
        return

    def __call__(self, dim: int):
        raise NotImplementedError


class AbstractPureState(AbstractState):
    r"""
    An abstract base class for all pure quantum states, equivalent to the state vector formalism.
    Pure states are associated with a Hilbert space of size,
    $\ket{\rho} \in \mathcal{H}^{d_1 \times \dots \times d_w}$
    and $w=$ `len(wires)` and $d$ is assigned at compile time.
    """

    def __init__(
        self,
        wires: Sequence[WiresTypes],
    ):
        super().__init__(wires=wires)
        return

    def __call__(self, dim: int):
        raise NotImplementedError


class AbstractMixedState(AbstractState):
    r"""
    An abstract base class for all mixed quantum states, equivalent to the density matrix formalism.
    Mixed states are associated with a Hilbert space of size,
    $\rho \in \mathcal{H}^{d_1 \times \dots \times d_w \times d_1 \times \dots \times d_w}$
    and $w=$ `len(wires)` and $d$ is assigned at compile time.
    """

    def __init__(
        self,
        wires: Sequence[WiresTypes],
    ):
        super().__init__(wires=wires)
        return

    def __call__(self, dim: int):
        raise NotImplementedError


class AbstractGate(AbstractOp):
    r"""
    An abstract base class for all unitary quantum gates, which transform an input state in a reversible way.
    $ U \in \mathcal{H}^{d_1 \times \dots \times d_w \times d_1 \times \dots \times d_w}$
    and $w=$ `len(wires)` and $d$ is assigned at compile time.
    """

    def __init__(
        self,
        wires: Sequence[WiresTypes],
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
        wires: Sequence[WiresTypes],
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
        wires: Sequence[WiresTypes],
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
        wires: Union[Sequence[WiresTypes], Sequence[Sequence[WiresTypes]]],
        where: Optional[Callable] = None,
        get: Optional[Callable] = None,
    ):
        # todo: handle the wires coming from both main and the shared wires
        copies = [eqx.tree_at(lambda op: op.wires, op, (wire,)) for wire in wires]
        self.copies = copies
        self.op = op

        if is_bearable(wires, Sequence[WiresTypes]):
            wires = op.wires + wires

        elif is_bearable(wires, Sequence[Sequence[WiresTypes]]):
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
    and $w=$ `len(wires)` and $d$ is assigned at compile time.
    """

    def __init__(
        self,
        wires: Sequence[WiresTypes],
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
    def __init__(self, wires: Sequence[WiresTypes]):
        super().__init__(wires=wires)
        return

    def __call__(self, dim: int):
        return None
