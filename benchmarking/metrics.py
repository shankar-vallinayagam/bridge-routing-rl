from typing import Iterable


def count_cx(circuit: Iterable[list]) -> int:
    return sum(gate[0] == "cx" for gate in circuit)


def count_named_gate(circuit: Iterable[list], name: str) -> int:
    return sum(gate[0].lower() == name.lower() for gate in circuit)


def validate_physical_circuit(circuit: Iterable[list], topology) -> None:
    adjacency = topology.adjacency
    for gate in circuit:
        if gate[0] != "cx":
            continue
        a, b = gate[1]
        if b not in adjacency[a]:
            raise ValueError(f"non-local CNOT found in final circuit: ({a}, {b})")


def circuit_depth(circuit: Iterable[list], qubit_count: int) -> int:
    layer_end = [0] * qubit_count
    depth = 0
    for name, operands in circuit:
        if name != "cx":
            continue
        layer = max(layer_end[operands[0]], layer_end[operands[1]]) + 1
        layer_end[operands[0]] = layer
        layer_end[operands[1]] = layer
        depth = max(depth, layer)
    return depth
