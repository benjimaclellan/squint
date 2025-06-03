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
