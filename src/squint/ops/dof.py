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

import functools
import itertools
from collections import OrderedDict
from typing import Optional, Union, Literal
import copy 
import equinox as eqx
import jax.numpy as jnp
import scipy as sp
from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Callable, Sequence
from uuid import uuid4 



class AbstractDoF(eqx.Module):
    pass

class DV(AbstractDoF):
    pass

class CV(AbstractDoF):
    pass

class TimeBin(AbstractDoF):
    pass

class FreqBin(AbstractDoF):
    pass

class Spatial(AbstractDoF):
    pass


class Wire(eqx.Module):
    idx: int = 0
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
            dim (int): The dimension of the local Hilbert space (the same dimension across all wires).
            dof (AbstractDoF): Type of degree of freedom that this wire represents, can be used to enfore circuit validity.
            idx: Unique identifier for the wire. If not provided, a random unique identifier is used.
        """
        if dim < 2:
            raise ValueError("Dimension should be 2 or greater.")
        self.dim = dim
        self.dof = dof
        self.idx = idx if idx is not None else str(uuid4())
        
