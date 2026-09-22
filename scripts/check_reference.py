#!/usr/bin/env python3
"""Verify all distributed data and reconstruct the principal paper point estimates."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def rows(name):
    with (ROOT / "data/reference" / (name + ".csv")).open() as f:
        return list(csv.DictReader(f))


def rmse(records):
    return float(
        np.rad2deg(np.sqrt(np.mean([float(r["squared_angle_radians"]) for r in records])))
    )


def main():
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for rel, expected in manifest["files"].items():
        p = ROOT / rel
        assert p.stat().st_size == expected["bytes"], rel
        assert hashlib.sha256(p.read_bytes()).hexdigest() == expected["sha256"], rel
    checks = {}
    full = rows("main")
    controls = rows("decoder-controls")
    with (ROOT / "results/paper/decoder_table.csv").open() as f:
        for row in csv.DictReader(f):
            method = {
                "mirror-VC": "mirror-vc",
                "same-query-rank": "rank",
                "quadratic-VC": "quadratic-vc",
            }[row["decoder"]]
            for geo in ["staggered-cycle", "frames"]:
                value = rmse(
                    [r for r in full + controls if r["decoder"] == method and r["geometry"] == geo]
                )
                assert np.isclose(value, float(row[geo]), rtol=1e-10, atol=1e-10)
                checks[f"{geo}/{method}"] = value
    curve = rows("curvature")
    with (ROOT / "results/paper/curvature.csv").open() as f:
        for row in csv.DictReader(f):
            for col, method in [
                ("quadratic_VC", "quadratic-vc"),
                ("linear_RankSVM", "linear"),
                ("rank_weights", "rank"),
            ]:
                value = rmse(
                    [
                        r
                        for r in curve
                        if r["decoder"] == method
                        and r["geometry"] == "staggered-cycle"
                        and float(r["gamma"]) == float(row["gamma"])
                    ]
                )
                assert np.isclose(value, float(row[col]), rtol=1e-10, atol=1e-10)
                checks[f'gamma={row["gamma"]}/{method}'] = value
    mismatch = rows("mismatch")
    for method in ["hybrid-vc", "rank", "soft-quadratic"]:
        selected = [r for r in mismatch if r["decoder"] == method and float(r["strength"]) == 0.03]
        checks["3%-cubic/" + method] = rmse(selected)
        assert len(selected) == 192
        failed = [r for r in selected if r["hard_success"] == "False"]
        assert len(failed) == 65
        checks["3%-failed/" + method] = rmse(failed)
    assert np.isclose(checks["3%-cubic/hybrid-vc"], 1.8233727815626108)
    ensemble = np.load(ROOT / "data/reference/ambiguity.npz", allow_pickle=False)
    assert ensemble["separated"].all()
    cycle = np.rad2deg(np.sqrt(ensemble["cycle_MSE"][:, :, 1].mean(1)))
    two = np.rad2deg(np.sqrt(ensemble["two_cycle_MSE"][:, :, 1].mean(1)))
    checks["ambiguity/cycle-median-pair-RMSE"] = float(np.median(cycle))
    checks["ambiguity/two-cycle-median-pair-RMSE"] = float(np.median(two))
    assert np.isclose(np.median(cycle), 4.382540082467466)
    assert np.isclose(np.median(two), 11.708990869618301)
    for file in (ROOT / "data/reference").glob("*.csv"):
        with file.open() as f:
            records = list(csv.DictReader(f))
        keys = [
            tuple(
                r[k] for k in ["target_id", "radius", "gamma", "strength", "geometry", "decoder"]
            )
            for r in records
        ]
        assert len(set(keys)) == len(keys), file.name
    print(
        json.dumps(
            dict(verified_files=len(manifest["files"]), point_estimates_degrees=checks), indent=2
        )
    )


if __name__ == "__main__":
    main()
