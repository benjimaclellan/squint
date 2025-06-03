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

import paramax
from beartype import beartype
from beartype.typing import Sequence
from jaxtyping import ArrayLike

from squint.ops.base import AbstractGate

__all__ = ["GlobalParameter"]


class GlobalParameter(AbstractGate):
    ops: Sequence[AbstractGate]
    weights: ArrayLike

    @beartype
    def __init__(
        self,
        ops: Sequence[AbstractGate],
        weights: ArrayLike,
    ):
        # if not len(ops) and weights.shape[0]
        wires = [wire for op in ops for wire in op.wires]
        super().__init__(wires=wires)
        self.ops = ops
        self.weights = paramax.non_trainable(weights)

    def unwrap(self):
        """Unwraps the shared ops for compilation and contractions."""
        return self.ops
