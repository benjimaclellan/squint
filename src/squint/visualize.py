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
import abc
import dataclasses
import itertools
from typing import Literal, Union

import matplotlib.pyplot as plt
from jax import numpy as jnp
from matplotlib.patches import Rectangle

from squint.circuit import Circuit
from squint.ops.base import (
    AbstractErasureChannel,
    AbstractGate,
    AbstractKrausChannel,
    AbstractMixedState,
    AbstractPureState,
)


@dataclasses.dataclass
class PlotConfig:
    wire_height: float = 1.0
    leg: float = 0.5
    vertical_width: float = 0.1
    width: float = 1.0
    height: float = 1.0


@dataclasses.dataclass
class WireData:
    wire: Union[int, str]
    y: float

    last_x: float = None
    next_y: float = None


class AbstractDiagramVisualizer(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def tensor_node(self):
        raise NotImplementedError

    @abc.abstractmethod
    def line(self):
        raise NotImplementedError

    @abc.abstractmethod
    def add_contraction(self):
        raise NotImplementedError

    @abc.abstractmethod
    def add_leg(self):
        raise NotImplementedError

    @abc.abstractmethod
    def add_channel(self):
        raise NotImplementedError


class MatplotlibDiagramVisualizer(AbstractDiagramVisualizer):
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect("equal")
        self.ax.axis("off")  # Hide axis

    def tensor_node(self, op, x, y, height=0.6, width=0.6, label: str = None):
        # Determine color based on type
        if isinstance(op, AbstractPureState):
            color = "orange"
        elif isinstance(op, AbstractMixedState):
            color = "red"
        elif isinstance(op, AbstractGate):
            color = "blue"
        elif isinstance(op, AbstractKrausChannel):
            color = "green"
        elif isinstance(op, AbstractErasureChannel):
            color = "purple"
        else:
            color = "gray"

        # Draw rectangle centered at (x, y)
        rect = Rectangle(
            (x - width / 2, y - height / 2),
            width,
            height,
            facecolor=f"{color}",
            edgecolor="black",
            linewidth=1,
            zorder=2,
        )
        self.ax.add_patch(rect)

        if label:
            self.ax.text(x, y, label, ha="center", va="center", fontsize=10)

        # Return a structure with a `.center` attribute like TikzPy
        class DummyRect:
            def __init__(self, center):
                self.center = center

        return DummyRect((x, y))

    def line(self, start, end, options):
        linestyle = "-"
        color = "black"

        if "dotted" in options:
            linestyle = ":"

        self.ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            linestyle=linestyle,
            color=color,
            zorder=1,
        )

    def add_contraction(self, start, end):
        self.line(start, end, options="dotted")

    def add_leg(self, start, end):
        self.line(start, end, options="")

    def add_channel(self, start, end):
        self.line(start, end, options="")

    def show(self):
        self.fig.tight_layout()
        self.fig.show()


class TikzDiagramVisualizer(AbstractDiagramVisualizer):
    def __init__(self):
        from tikzpy import TikzPicture

        self.fig = TikzPicture(center=True)  # Initialize empty fig

    def tensor_node(self, op, x, y, height=0.6, width=0.6, label: str = None):
        if isinstance(op, AbstractPureState):
            color = "orange"
        elif isinstance(op, AbstractMixedState):
            color = "red"
        elif isinstance(op, AbstractGate):
            color = "blue"
        elif isinstance(op, AbstractKrausChannel):
            color = "green"
        elif isinstance(op, AbstractErasureChannel):
            color = "purple"

        tensor = self.fig.rectangle_from_center(
            (x, y), height=height, width=width, options=f"draw=none,fill={color}!15"
        )
        if label:
            self.fig.node(tensor.center, text=f"{label}")
        return tensor

    def line(self, start, end, options):
        self.fig.line(
            start=start,
            end=end,
            options=options,
        )

    def add_contraction(self, start, end):
        self.line(start, end, options="dotted")

    def add_leg(self, start, end):
        self.line(start, end, options="")

    def add_channel(self, start, end):
        self.line(start, end, options="")


def draw(circuit: Circuit, drawer: Literal["mpl", "tikz"] = "mpl"):
    """
    Circuit diagram visualizer.
    Draws a circuit as a tensor network diagram using either Matplotlib or TikZ.

    Args:
        circuit (Circuit): The circuit to visualize.
        drawer (str): The visualization backend to use, either "mpl" for Matplotlib or "tikz" for TikZ.
    """
    if drawer == "tikz":
        drawer = TikzDiagramVisualizer()

    elif drawer == "mpl":
        drawer = MatplotlibDiagramVisualizer()

    config = PlotConfig(wire_height=1.0, width=0.5, height=0.5, vertical_width=0.2)
    wire_data = {
        wire: WireData(wire=wire, y=i * config.wire_height, last_x=0.0)
        for i, wire in enumerate(sorted(circuit.wires))
    }

    backend = circuit.backend

    iterator_channel_ind = itertools.count(1)
    for i, (key, _op) in enumerate(circuit.ops.items(), start=1):
        for op in _op.unwrap():
            x = i * config.wire_height  # TODO:
            label = key

            # multi-wire connection vertically
            if len(op.wires) > 1:
                y_max = max([wire_data[wire].y for wire in op.wires])
                y_min = min([wire_data[wire].y for wire in op.wires])
                height = y_max - y_min
                y = (y_max + y_min) / 2

                drawer.tensor_node(op, x, y, height=height, width=config.vertical_width)
                if backend == "mixed":
                    drawer.tensor_node(
                        op, -x, y, height=height, width=config.vertical_width
                    )

            for wire in op.wires:
                drawer.add_leg(
                    start=(x, wire_data[wire].y),
                    end=(x + config.leg, wire_data[wire].y),
                )
                if backend == "mixed":
                    drawer.add_leg(
                        start=(-x, wire_data[wire].y),
                        end=(-x - config.leg, wire_data[wire].y),
                    )

                if isinstance(
                    op, (AbstractGate, AbstractKrausChannel, AbstractErasureChannel)
                ):
                    drawer.add_leg(
                        start=(x, wire_data[wire].y),
                        end=(x - config.leg, wire_data[wire].y),
                    )
                    drawer.add_contraction(
                        start=(x - config.leg, wire_data[wire].y),
                        end=(wire_data[wire].last_x, wire_data[wire].y),
                    )
                    if backend == "mixed":
                        drawer.add_leg(
                            start=(-x, wire_data[wire].y),
                            end=(-x + config.leg, wire_data[wire].y),
                        )
                        drawer.add_contraction(
                            start=(-x + config.leg, wire_data[wire].y),
                            end=(-wire_data[wire].last_x, wire_data[wire].y),
                        )

                if isinstance(op, AbstractKrausChannel):
                    channel_height = next(iterator_channel_ind) * config.wire_height

                    drawer.add_leg(
                        start=(x, wire_data[wire].y),
                        end=(x, wire_data[wire].y - config.leg),
                    )
                    drawer.add_leg(
                        start=(-x, wire_data[wire].y),
                        end=(-x, wire_data[wire].y - config.leg),
                    )
                    lines = [
                        (x, wire_data[wire].y - config.leg),
                        (x, -channel_height),
                        (-x, -channel_height),
                        (-x, wire_data[wire].y - config.leg),
                    ]
                    for k in range(len(lines) - 1):
                        drawer.add_channel(start=lines[k], end=lines[k + 1])

                if isinstance(op, AbstractErasureChannel):
                    channel_height = next(iterator_channel_ind) * config.wire_height
                    lines = [
                        (x + config.leg, wire_data[wire].y),
                        (x + 2 * config.leg, wire_data[wire].y),
                        (x + 2 * config.leg, -channel_height),
                        (-x - 2 * config.leg, -channel_height),
                        (-x - 2 * config.leg, wire_data[wire].y),
                        (-x - config.leg, wire_data[wire].y),
                    ]
                    for k in range(len(lines) - 1):
                        drawer.add_leg(
                            start=lines[k],
                            end=lines[k + 1],  # options=options["channel"]
                        )

                wire_data[wire].last_x = x + config.leg

                if isinstance(op, AbstractMixedState):
                    drawer.tensor_node(
                        op,
                        0.0,
                        wire_data[wire].y,
                        height=config.vertical_width,
                        width=2 * x,
                    )

                drawer.tensor_node(
                    op,
                    x,
                    wire_data[wire].y,
                    height=config.height,
                    width=config.height,
                    label=label,
                )

                if backend == "mixed":
                    drawer.tensor_node(
                        op,
                        -x,
                        wire_data[wire].y,
                        height=config.height,
                        width=config.height,
                    )

    return drawer.fig


# %%
if __name__ == "__main__":
    # %%
    import itertools

    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from rich.pretty import pprint

    from squint.circuit import Circuit
    from squint.ops.dv import DiscreteVariableState, HGate, RZGate

    circuit = Circuit(backend="pure")

    circuit.add(DiscreteVariableState(wires=(0,), n=(0,)))
    circuit.add(HGate(wires=(0,)))
    circuit.add(RZGate(wires=(0,), phi=0.0 * jnp.pi), "phase")
    circuit.add(HGate(wires=(0,)))

    pprint(circuit)

    fig = draw(circuit, drawer="mpl")
    fig.show()

