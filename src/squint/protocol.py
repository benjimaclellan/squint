# %%

import equinox as eqx
from beartype import beartype

from squint.circuit import Circuit
from squint.estimator import Estimator

# TODO: build out Protocol abstraction
class Protocol(eqx.Module):
    circuit: Circuit
    estimator: Estimator

    @beartype
    def __init__(
        self,
    ):
        """ """
        pass
