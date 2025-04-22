import sys

# from loguru import logger
# logger.disable()
from loguru import logger as log

from squint.ops.base import (
    AbstractErasureChannel,
    AbstractGate,
    AbstractKrausChannel,
    AbstractMeasurement,
    AbstractMixedState,
    AbstractOp,
    AbstractPureState,
    create,
    destroy,
)

log.remove()  # remove the old handler. Else, the old one will work along with the new one you've added below'
log.add(sys.stderr, level="INFO")


__all__ = [
    "AbstractOp",
    "AbstractGate",
    "AbstractMeasurement",
    "AbstractPureState",
    "AbstractMixedState",
    "AbstractKrausChannel",
    "AbstractErasureChannel",
    "create",
    "destroy",
]
