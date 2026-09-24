import csv
import json
from pathlib import Path

from benchmarking.reporting import write_summaries


def run_benchmark(cases, adapters, output_dir: Path, seeds=(1,)):
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for case in cases:
        for adapter in adapters:
            for seed in seeds:
                result = adapter.route(case, seed=seed)
                row = result.to_dict()
                row["seed"] = seed
                row["method"] = adapter.method_name
                row["case_seed"] = case.seed
                row["depth_bucket"] = case.metadata.get("depth_bucket")
                results.append(row)

    with (output_dir / "results.json").open("w") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")
    fields = sorted({key for row in results for key in row})
    with (output_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    write_summaries(results, output_dir)
    return results
