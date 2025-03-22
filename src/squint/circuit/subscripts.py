def subscripts(circuit, get_symbol, get_symbol_channel):
    _iterator = itertools.count(0)
    _iterator_channel = itertools.count(0)

    _in_axes = []
    _out_axes = []
    _wire_chars = {wire: [] for wire in circuit.wires}
    for op in circuit.unwrap():
        _axis = []
        for wire in op.wires:
            if isinstance(op, AbstractState):
                _in_axis = ""
                _out_axes = get_symbol(next(_iterator))

            elif isinstance(op, AbstractGate):
                _in_axis = _wire_chars[wire][-1]
                _out_axes = get_symbol(next(_iterator))

            elif isinstance(op, AbstractChannel):
                _in_axis = _wire_chars[wire][-1]
                _out_axes = get_symbol(next(_iterator))

            elif isinstance(op, AbstractMeasurement):
                _in_axis = _wire_chars[wire][-1]
                _right_axis = ""

            else:
                raise TypeError

            _axis += [_in_axis, _out_axes]

            _wire_chars[wire].append(_out_axes)

        # add extra axis for channel
        if isinstance(op, AbstractChannel):
            _axis.insert(0, get_symbol_channel(next(_iterator_channel)))

        _in_axes.append("".join(_axis))

    _out_axes = [val[-1] for key, val in _wire_chars.items()]

    _in_expr = ",".join(_in_axes)
    _out_expr = "".join(_out_axes)
    _subscripts = f"{_in_expr}->{_out_expr}"
    return _in_expr, _out_expr


_in_expr_ket, _out_expr_ket = subscripts(circuit, get_symbol_right, get_symbol_channel)
_in_expr_bra, _out_expr_bra = subscripts(circuit, get_symbol_left, get_symbol_channel)

_tensors_ket = [op(dim=dim) for op in circuit.unwrap()]
_tensors_bra = [op(dim=dim).conj() for op in circuit.unwrap()]


print(_in_expr_ket, _out_expr_ket)
print(_in_expr_bra, _out_expr_bra)
_subscripts = f"{_in_expr_ket},{_in_expr_bra}->{_out_expr_ket}{_out_expr_bra}"
print(_subscripts)
