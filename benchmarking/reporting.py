import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def write_summaries(results, output_dir: Path) -> None:
    groups = defaultdict(list)
    depth_groups = defaultdict(list)
    for result in results:
        groups[result["method"]].append(result)
        depth_groups[(result["method"], result.get("depth_bucket"))].append(result)

    summary_rows = [_summarize(method, rows) for method, rows in sorted(groups.items())]
    depth_rows = [
        dict(_summarize(method, rows), depth_bucket=depth)
        for (method, depth), rows in sorted(depth_groups.items())
    ]
    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_csv(output_dir / "summary_by_depth.csv", depth_rows)


def _summarize(method, rows):
    successful = [row for row in rows if row["success"]]
    added = [row["added_cnot_count"] for row in successful if row["added_cnot_count"] is not None]
    times = [row["route_time_seconds"] for row in successful if row["route_time_seconds"] is not None]
    inference = [
        row["inference_time_seconds"]
        for row in successful
        if row.get("inference_time_seconds") is not None
    ]
    return {
        "method": method,
        "case_count": len(rows),
        "successful_cases": len(successful),
        "success_rate": len(successful) / len(rows) if rows else 0.0,
        "mean_added_cnot": mean(added) if added else None,
        "median_added_cnot": median(added) if added else None,
        "mean_route_time_seconds": mean(times) if times else None,
        "mean_inference_time_seconds": mean(inference) if inference else None,
    }


def _write_csv(path: Path, rows):
    if not rows:
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
