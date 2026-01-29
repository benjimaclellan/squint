# Tests for visualization module

import matplotlib
import matplotlib.pyplot as plt

from squint.circuit import Circuit
from squint.ops.base import SharedGate, Wire
from squint.ops.dv import (
    CXGate,
    DiscreteVariableState,
    HGate,
    RZGate,
)
from squint.ops.noise import DepolarizingChannel
from squint.visualize import (
    MatplotlibDiagramVisualizer,
    PlotConfig,
    WireData,
    draw,
)

# Use non-interactive backend for tests
matplotlib.use("Agg")


class TestPlotConfig:
    def test_default_config(self):
        """Test PlotConfig has correct default values."""
        config = PlotConfig()
        assert config.wire_height == 1.0
        assert config.leg == 0.5
        assert config.vertical_width == 0.1
        assert config.width == 1.0
        assert config.height == 1.0

    def test_custom_config(self):
        """Test PlotConfig accepts custom values."""
        config = PlotConfig(wire_height=2.0, leg=0.3, width=1.5)
        assert config.wire_height == 2.0
        assert config.leg == 0.3
        assert config.width == 1.5


class TestWireData:
    def test_wire_data_creation(self):
        """Test WireData stores wire information correctly."""
        wire_data = WireData(wire=0, y=1.0, last_x=0.5)
        assert wire_data.wire == 0
        assert wire_data.y == 1.0
        assert wire_data.last_x == 0.5

    def test_wire_data_defaults(self):
        """Test WireData has correct default values."""
        wire_data = WireData(wire="q0", y=2.0)
        assert wire_data.wire == "q0"
        assert wire_data.y == 2.0
        assert wire_data.last_x is None
        assert wire_data.next_y is None


class TestMatplotlibDiagramVisualizer:
    def test_visualizer_creation(self):
        """Test MatplotlibDiagramVisualizer creates fig and ax."""
        viz = MatplotlibDiagramVisualizer()
        assert viz.fig is not None
        assert viz.ax is not None
        plt.close(viz.fig)

    def test_tensor_node_returns_object_with_center(self):
        """Test tensor_node returns object with center attribute."""
        viz = MatplotlibDiagramVisualizer()
        wire = Wire(dim=2, idx=0)
        op = DiscreteVariableState(wires=(wire,), n=(0,))

        result = viz.tensor_node(op, x=1.0, y=2.0)
        assert hasattr(result, "center")
        assert result.center == (1.0, 2.0)
        plt.close(viz.fig)


class TestDraw:
    def test_draw_single_qubit_circuit(self):
        """Test drawing a simple single-qubit circuit."""
        wire = Wire(dim=2, idx=0)
        circuit = Circuit()
        circuit.add(DiscreteVariableState(wires=(wire,), n=(0,)))
        circuit.add(HGate(wires=(wire,)))

        fig = draw(circuit, drawer="mpl")
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_draw_multi_qubit_circuit(self):
        """Test drawing a multi-qubit circuit with entangling gates."""
        wire0 = Wire(dim=2, idx=0)
        wire1 = Wire(dim=2, idx=1)

        circuit = Circuit()
        circuit.add(DiscreteVariableState(wires=(wire0,), n=(0,)))
        circuit.add(DiscreteVariableState(wires=(wire1,), n=(0,)))
        circuit.add(HGate(wires=(wire0,)))
        circuit.add(CXGate(wires=(wire0, wire1)))

        fig = draw(circuit, drawer="mpl")
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_draw_circuit_with_parameterized_gate(self):
        """Test drawing a circuit with parameterized gates."""
        wire = Wire(dim=2, idx=0)
        circuit = Circuit()
        circuit.add(DiscreteVariableState(wires=(wire,), n=(0,)))
        circuit.add(HGate(wires=(wire,)))
        circuit.add(RZGate(wires=(wire,), phi=0.5), "phase")
        circuit.add(HGate(wires=(wire,)))

        fig = draw(circuit, drawer="mpl")
        assert fig is not None
        plt.close(fig)

    def test_draw_circuit_with_noise(self):
        """Test drawing a circuit with noise channels."""
        wire = Wire(dim=2, idx=0)
        circuit = Circuit()
        circuit.add(DiscreteVariableState(wires=(wire,), n=(0,)))
        circuit.add(HGate(wires=(wire,)))
        circuit.add(DepolarizingChannel(wires=(wire,), p=0.1))

        fig = draw(circuit, drawer="mpl")
        assert fig is not None
        plt.close(fig)

    def test_draw_ghz_circuit(self):
        """Test drawing a GHZ state preparation circuit."""
        n = 3
        wires = [Wire(dim=2, idx=i) for i in range(n)]

        circuit = Circuit()
        for w in wires:
            circuit.add(DiscreteVariableState(wires=(w,), n=(0,)))

        circuit.add(HGate(wires=(wires[0],)))
        for i in range(n - 1):
            circuit.add(CXGate(wires=(wires[i], wires[i + 1])))

        circuit.add(
            SharedGate(op=RZGate(wires=(wires[0],), phi=0.0), wires=tuple(wires[1:])),
            "phase",
        )

        fig = draw(circuit, drawer="mpl")
        assert fig is not None
        plt.close(fig)

    def test_draw_returns_figure(self):
        """Test that draw returns a matplotlib Figure object."""
        wire = Wire(dim=2, idx=0)
        circuit = Circuit()
        circuit.add(DiscreteVariableState(wires=(wire,), n=(0,)))

        result = draw(circuit, drawer="mpl")
        assert isinstance(result, plt.Figure)
        plt.close(result)
