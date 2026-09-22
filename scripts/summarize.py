#!/usr/bin/env python3
"""Summarize a replay file and match losses by target identity, never row order."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["target_id", "radius", "gamma", "strength", "geometry", "decoder"]


def key(r):
    return tuple(float(r[k]) if k in ["radius", "gamma", "strength"] else r[k] for k in KEYS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    args = parser.parse_args()
    references = {}
    for p in (ROOT / "data/reference").glob("*.csv"):
        with p.open() as f:
            for r in csv.DictReader(f):
                references[key(r)] = float(r["squared_angle_radians"])
    groups = defaultdict(list)
    ambiguity = defaultdict(list)
    ambiguity_reference = np.load(ROOT / "data/reference/ambiguity.npz", allow_pickle=False)
    seen = set()
    differences = []
    failures = []
    completed = []
    for path in args.files:
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                if r["kind"] == "failure":
                    failures.append(r)
                elif r["kind"] == "completion":
                    completed.append(r)
                elif r["kind"] == "ambiguity":
                    identity = (r["geometry"], r["pair"])
                    if identity in seen:
                        raise ValueError(f"duplicate ambiguity pair: {identity}")
                    seen.add(identity)
                    ambiguity[r["geometry"]].append(r)
                    field = "cycle_MSE" if r["geometry"] == "staggered-cycle" else "two_cycle_MSE"
                    expected = ambiguity_reference[field][r["pair"], :, 1]
                    differences.extend(
                        np.abs(np.asarray(r["squared_angle_radians"]) - expected).tolist()
                    )
                elif r["kind"] == "result":
                    k = key(r) + (r.get("C"),)
                    if k in seen:
                        raise ValueError(f"duplicate result across shards: {k}")
                    seen.add(k)
                    groups[
                        (
                            r["geometry"],
                            r["decoder"],
                            r["radius"],
                            r["gamma"],
                            r["strength"],
                            r.get("C"),
                        )
                    ].append(r)
                    if key(r) in references and r.get("C") is None:
                        differences.append(abs(r["squared_angle_radians"] - references[key(r)]))
    if len(completed) != len(args.files):
        raise ValueError("one or more replay files are incomplete")
    summary = []
    for k, records in groups.items():
        angles = np.rad2deg(np.sqrt([r["squared_angle_radians"] for r in records]))
        summary.append(
            dict(
                geometry=k[0],
                decoder=k[1],
                radius=k[2],
                gamma=k[3],
                strength=k[4],
                C=k[5],
                targets=len(records),
                RMSE_degrees=float(np.sqrt(np.mean(angles**2))),
                median_degrees=float(np.median(angles)),
                p90_degrees=float(np.quantile(angles, 0.9)),
            )
        )
    pair_summary = []
    for geometry, records in ambiguity.items():
        pair_rmse = np.sqrt([np.mean(r["squared_angle_radians"]) for r in records])
        bounds = np.asarray([r["disconnected_bound_radians"] for r in records])
        pair_summary.append(
            dict(
                geometry=geometry,
                pairs=len(records),
                separation_rate=float(np.mean([not r["same_order"] for r in records])),
                median_pair_RMSE_degrees=float(np.rad2deg(np.median(pair_rmse))),
                fraction_below_pair_specific_bound=float(np.mean(pair_rmse < bounds)),
            )
        )
    print(
        json.dumps(
            dict(
                groups=summary,
                ambiguity=pair_summary,
                matched_reference_rows=len(differences),
                maximum_squared_radian_difference=max(differences, default=None),
                failures=failures,
            ),
            indent=2,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
