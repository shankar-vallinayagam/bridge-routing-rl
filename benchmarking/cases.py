from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


Gate = Tuple[str, Tuple[int, int]]


@dataclass(frozen=True)
class BenchmarkCase:
    """One immutable routing problem shared by every benchmark adapter."""

    case_id: str
    qubit_count: int
    gates: Tuple[Gate, ...]
    seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def original_cnot_count(self) -> int:
        return sum(name == "cx" for name, _ in self.gates)

    def as_circuit_list(self) -> List[list]:
        return [[name, list(operands)] for name, operands in self.gates]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "qubit_count": self.qubit_count,
            "gates": [[name, list(operands)] for name, operands in self.gates],
            "seed": self.seed,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, record: Dict[str, Any]) -> "BenchmarkCase":
        gates = tuple(
            (name, tuple(int(qubit) for qubit in operands))
            for name, operands in record["gates"]
        )
        return cls(
            case_id=record["case_id"],
            qubit_count=int(record["qubit_count"]),
            gates=gates,
            seed=record.get("seed"),
            metadata=dict(record.get("metadata", {})),
        )


def validate_case(case: BenchmarkCase) -> None:
    if case.qubit_count < 2:
        raise ValueError("a benchmark case needs at least two qubits")
    for name, operands in case.gates:
        if name != "cx" or len(operands) != 2:
            raise ValueError("benchmark cases currently support only two-qubit cx gates")
        if any(qubit < 0 or qubit >= case.qubit_count for qubit in operands):
            raise ValueError(f"gate in {case.case_id} refers to an invalid qubit")
        if operands[0] == operands[1]:
            raise ValueError(f"gate in {case.case_id} contains a self-loop")
