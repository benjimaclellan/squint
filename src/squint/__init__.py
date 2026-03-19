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

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

# Re-export Circuit at the top level for convenience.
from squint.interface.base import Circuit, Block, SharedGate, Wire  # noqa: E402

# Eagerly import op modules so that AbstractProcess._registry is fully populated
# whenever squint is imported (e.g., for ops_for_backend discovery in tests).
import squint.interface.dv  # noqa: F401, E402
import squint.interface.fock  # noqa: F401, E402
import squint.interface.noise  # noqa: F401, E402

__all__ = ["Circuit"]
