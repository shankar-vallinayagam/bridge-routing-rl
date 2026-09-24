import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from env.chip_architecture import ChipHardware


@dataclass(frozen=True)
class TopologySpec:
    topology_id: str
    qubit_count: int
    edges: Tuple[Tuple[int, int], ...]

    @classmethod
    def from_dict(cls, record):
        return cls(
            topology_id=record["topology_id"],
            qubit_count=int(record["qubit_count"]),
            edges=tuple(tuple(int(q) for q in edge) for edge in record["edges"]),
        )

    @classmethod
    def from_json(cls, path: Path) -> "TopologySpec":
        with path.open() as handle:
            return cls.from_dict(json.load(handle))

    @property
    def adjacency(self) -> List[List[int]]:
        adjacency = [[] for _ in range(self.qubit_count)]
        for a, b in self.edges:
            adjacency[a].append(b)
            adjacency[b].append(a)
        return adjacency

    def chip_hardware(self) -> ChipHardware:
        return ChipHardware(self.qubit_count, self.adjacency)
