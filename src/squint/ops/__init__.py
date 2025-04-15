import sys

# from loguru import logger
# logger.disable()
from loguru import logger as log

from squint.ops.base import (
    AbstractGate,
    AbstractMeasurement,
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
    "AbstractState",
    "create",
    "destroy",
]
