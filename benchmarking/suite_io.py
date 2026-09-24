import json
from pathlib import Path
from typing import Iterable, List

from benchmarking.cases import BenchmarkCase, validate_case


def load_cases(path: Path) -> List[BenchmarkCase]:
    cases = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                case = BenchmarkCase.from_dict(json.loads(line))
                validate_case(case)
            except Exception as exc:
                raise ValueError(f"invalid benchmark case at {path}:{line_number}: {exc}") from exc
            cases.append(case)
    if not cases:
        raise ValueError(f"benchmark suite is empty: {path}")
    return cases


def write_cases(path: Path, cases: Iterable[BenchmarkCase], manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for case in cases:
            handle.write(json.dumps(case.to_dict(), sort_keys=True) + "\n")
    with (path.parent / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
