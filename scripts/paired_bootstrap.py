#!/usr/bin/env python3
"""Target-paired, stratified GPU bootstrap for the paper's acquisition contrasts."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--population",
        choices=["original-full", "original-matched", "independent-validation", "random-mirrors"],
        required=True,
    )
    parser.add_argument(
        "--reference", choices=["frames", "staggered-two", "random-mirrors"], required=True
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if (args.population == "random-mirrors") != (args.reference == "random-mirrors"):
        parser.error("random-mirrors requires the matching population and reference")
    torch.set_num_threads(1)
    torch.cuda.set_device(args.device)
    settings = json.loads((ROOT / "configs/paper.json").read_text())["statistics"]
    name = (
        "independent"
        if args.population == "independent-validation"
        else "random-mirrors" if args.population == "random-mirrors" else "main"
    )
    with (ROOT / f"data/reference/{name}.csv").open() as f:
        rows = list(csv.DictReader(f))
    matched = args.population in ["original-matched", "independent-validation"]
    values = {}
    for r in rows:
        if r["decoder"] != "quadratic-vc" or r["geometry"] not in [
            "staggered-cycle",
            args.reference,
        ]:
            continue
        if matched and (
            int(r["dimension"]) not in [10, 20] or float(r["radius"]) not in [1.0, 10.0]
        ):
            continue
        ident = tuple(int(r[k]) for k in ["replica", "stream", "regime", "dimension"]) + (
            int(r["target_id"].split("/")[-1]),
        )
        k = ident + (r["geometry"], float(r["radius"]))
        if k in values:
            raise ValueError("duplicate target identity")
        values[k] = float(r["squared_angle_radians"])
    identities = sorted({k[:5] for k in values})
    radii = [1.0, 10.0] if matched else [0.1, 1.0, 10.0]
    losses = []
    for geo in ["staggered-cycle", args.reference]:
        losses.append(
            torch.tensor(
                [[values[t + (geo, r)] for r in radii] for t in identities],
                device=args.device,
                dtype=torch.float64,
            )
        )
    candidate, reference = losses
    strata = defaultdict(list)
    for i, t in enumerate(identities):
        strata[(t[1], t[2], t[3])].append(i)
    offset = {
        "original-full": 10,
        "original-matched": 20,
        "independent-validation": 30,
        "random-mirrors": 3100,
    }[args.population]
    if args.reference == "frames":
        offset += 1
    rng = torch.Generator(device=args.device).manual_seed(settings["bootstrap_seed"] + offset)
    sampled = []
    for s in sorted(strata):
        positions = torch.tensor(strata[s], device=args.device)
        sampled.append(
            positions[
                torch.randint(
                    len(positions),
                    (settings["bootstrap_draws"], len(positions)),
                    generator=rng,
                    device=args.device,
                )
            ]
        )
    sample = torch.cat(sampled, 1)
    c, r = candidate.mean(1), reference.mean(1)
    improvements = torch.cat([1 - c[x].mean(1) / r[x].mean(1) for x in sample.split(256)])
    confidence = 0.95 if args.reference == "random-mirrors" else 0.975
    tail = (1 - confidence) / 2
    interval = torch.quantile(
        improvements, torch.tensor([tail, 1 - tail], device=args.device, dtype=torch.float64)
    )
    print(
        json.dumps(
            dict(
                population=args.population,
                reference=args.reference,
                targets=len(identities),
                relative_MSE_reduction=float(1 - c.mean() / r.mean()),
                confidence=confidence,
                interval=interval.cpu().tolist(),
                target_win_rate=float((c < r).double().mean()),
                bootstrap_draws=settings["bootstrap_draws"],
                seed=settings["bootstrap_seed"] + offset,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
