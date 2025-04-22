import abc
import dataclasses
import itertools
from typing import Union

from tikzpy import TikzPicture

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


# TODO: add matplotlib and Tikz backends, using an ABC Drawer and generalized method
class AbstractDiagramVisualizer(abc.ABC):
    @abc.abstractclassmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def add_node(self):
        raise NotImplementedError

    @abc.abstractmethod
    def add_line(self):
        raise NotImplementedError

    @abc.abstractmethod
    def add_leg(self):
        raise NotImplementedError


class MatplotlibDiagramVisualizer(AbstractDiagramVisualizer):
    pass


class TikzDiagramVisualizer(AbstractDiagramVisualizer):
    pass


def draw(circuit: Circuit):
    tikz = TikzPicture(center=True)  # Initialize empty canvas
    config = PlotConfig(wire_height=1.0, width=0.5, height=0.5, vertical_width=0.2)
    wire_data = {
        wire: WireData(wire=wire, y=i * config.wire_height, last_x=0.0)
        for i, wire in enumerate(circuit.wires)
    }

    backend = circuit.backend

    options = {
        "leg": "",
        "contraction": "dotted",
        "channel": "dashed",
    }

    def tensor_node(op, x, y, height=0.6, width=0.6):
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

        tensor = tikz.rectangle_from_center(
            (x, y), height=height, width=width, options=f"draw=none,fill={color}!15"
        )
        # tensor = tikz.circle((x, y), width/4, options=f"thin, fill={color}!15")
        return tensor

    # kraus_height = len(wire_data.keys()) * config.wire_height

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

                tensor = tensor_node(
                    op, x, y, height=height, width=config.vertical_width
                )
                if backend == "mixed":
                    tensor = tensor_node(
                        op, -x, y, height=height, width=config.vertical_width
                    )

            for wire in op.wires:
                tikz.line(
                    start=(x, wire_data[wire].y),
                    end=(x + config.leg, wire_data[wire].y),
                    options=options["leg"],
                )
                if backend == "mixed":
                    tikz.line(
                        start=(-x, wire_data[wire].y),
                        end=(-x - config.leg, wire_data[wire].y),
                        options=options["leg"],
                    )

                if isinstance(
                    op, (AbstractGate, AbstractKrausChannel, AbstractErasureChannel)
                ):
                    tikz.line(
                        start=(x, wire_data[wire].y),
                        end=(x - config.leg, wire_data[wire].y),
                        options=options["leg"],
                    )
                    tikz.line(
                        start=(x - config.leg, wire_data[wire].y),
                        end=(wire_data[wire].last_x, wire_data[wire].y),
                        options=options["contraction"],
                    )
                    if backend == "mixed":
                        tikz.line(
                            start=(-x, wire_data[wire].y),
                            end=(-x + config.leg, wire_data[wire].y),
                            options=options["leg"],
                        )
                        tikz.line(
                            start=(-x + config.leg, wire_data[wire].y),
                            end=(-wire_data[wire].last_x, wire_data[wire].y),
                            options=options["contraction"],
                        )

                if isinstance(op, AbstractKrausChannel):
                    channel_height = next(iterator_channel_ind) * config.wire_height

                    tikz.line(
                        start=(x, wire_data[wire].y),
                        end=(x, wire_data[wire].y - config.leg),
                        options=options["leg"],
                    )
                    tikz.line(
                        start=(-x, wire_data[wire].y),
                        end=(-x, wire_data[wire].y - config.leg),
                        options=options["leg"],
                    )
                    lines = [
                        (x, wire_data[wire].y - config.leg),
                        (x, -channel_height),
                        (-x, -channel_height),
                        (-x, wire_data[wire].y - config.leg),
                    ]
                    for k in range(len(lines) - 1):
                        tikz.line(
                            start=lines[k], end=lines[k + 1], options=options["channel"]
                        )

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
                        tikz.line(
                            start=lines[k], end=lines[k + 1], options=options["channel"]
                        )

                wire_data[wire].last_x = x + config.leg

                if isinstance(op, AbstractMixedState):
                    tensor = tensor_node(
                        op,
                        0.0,
                        wire_data[wire].y,
                        height=config.vertical_width,
                        width=2 * x,
                    )

                tensor = tensor_node(
                    op, x, wire_data[wire].y, height=config.height, width=config.height
                )
                tikz.node(tensor.center, text=f"{label}")

                if backend == "mixed":
                    tensor = tensor_node(
                        op,
                        -x,
                        wire_data[wire].y,
                        height=config.height,
                        width=config.height,
                    )

    return tikz
