#!/usr/bin/env python3
"""Replay frozen paper inputs with the released solvers and ordinal-only API."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from connected_query_designs import make_queries, recover_direction
from connected_query_designs.api import scores_from_order
from connected_query_designs import decoders

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT / "configs/paper.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        choices=[
            "main",
            "independent",
            "curvature",
            "random-mirrors",
            "mismatch",
            "ambiguity",
            "linear-calibration",
        ],
    )
    parser.add_argument(
        "--device", default="cuda:0", help="Logical device within CUDA_VISIBLE_DEVICES"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New JSONL file; existing files are never overwritten",
    )
    parser.add_argument("--dimension", type=int, choices=[8, 10, 20, 50])
    parser.add_argument(
        "--max-groups", type=int, help="Smoke-test subset; omit for the full cohort"
    )
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard_index < args.shards or args.threads < 1:
        parser.error("invalid shard or thread count")
    if args.max_groups is not None and args.max_groups < 1:
        parser.error("max-groups must be positive")
    if not torch.cuda.is_available():
        parser.error("a visible CUDA GPU is required for solver replay")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(args.device)
    device = torch.device(args.device)
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    settings = config[args.experiment]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def load(relative):
        p = ROOT / relative
        if hashlib.sha256(p.read_bytes()).hexdigest() != manifest["files"][relative]["sha256"]:
            raise ValueError(f"input checksum mismatch: {relative}")
        with np.load(p, allow_pickle=False) as f:
            return {k: torch.as_tensor(f[k], device=device) for k in f.files}

    def order(points, gradient, hessian):
        return decoders.quadratic_scores(points, gradient, hessian).argsort(dim=-1, stable=True)

    failures = 0
    with args.output.open("x") as output:

        def emit(record):
            output.write(json.dumps(record, allow_nan=False) + "\n")
            output.flush()

        emit(
            dict(
                kind="environment",
                experiment=args.experiment,
                device=str(device),
                gpu=torch.cuda.get_device_name(device),
                torch=torch.__version__,
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                cpu_affinity=(
                    sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
                ),
                threads=args.threads,
                shards=args.shards,
                shard_index=args.shard_index,
                subset=args.max_groups is not None or args.dimension is not None,
            )
        )

        def record(
            group,
            geometry,
            method,
            direction,
            truth,
            *,
            radius=1.0,
            gamma=1.0,
            strength=0.0,
            hard=None,
            elapsed=None,
            C=None,
        ):
            losses = decoders.angular_mse(direction, truth).cpu().tolist()
            for i, loss in enumerate(losses):
                emit(
                    dict(
                        kind="result",
                        cohort=group["cohort"],
                        target_id=f"{group['cohort']}/{group['id']}/{i}",
                        **{k: group[k] for k in ["replica", "stream", "regime", "dimension"]},
                        radius=radius,
                        gamma=gamma,
                        strength=strength,
                        geometry=geometry,
                        decoder=method,
                        squared_angle_radians=loss,
                        certified=True,
                        hard_success=bool(hard[i]) if hard is not None else None,
                        batch_seconds=elapsed,
                        C=C,
                    )
                )

        def centers(group, points, ranking, truth, geometry, **extra):
            start = time.perf_counter()
            result, info = decoders.quadratic_centers(points, scores_from_order(points, ranking))
            if not bool(result["certified"].all()):
                raise ArithmeticError("uncertified center")
            elapsed = time.perf_counter() - start
            for j, method in enumerate(["quadratic-ac", "quadratic-vc"]):
                record(
                    group,
                    geometry,
                    method,
                    result["direction"][:, j],
                    truth,
                    elapsed=elapsed,
                    **extra,
                )
            return result

        def run(group):
            data = load(group["file"])
            gradient, hessian = data["gradient"], data["hessian"]
            dimension = group["dimension"]

            def queries(geometry, radius=1.0):
                return make_queries(
                    dimension,
                    batch_size=len(gradient),
                    bases=data["bases"],
                    geometry=geometry,
                    radius=radius,
                    device=device,
                )

            if args.experiment in ("main", "independent"):
                for geometry in settings["geometries"]:
                    for radius in settings["radii"]:
                        points = queries(geometry, radius)
                        ranking = order(points, gradient, hessian)
                        centers(group, points, ranking, gradient, geometry, radius=radius)
                        if args.experiment == "main" and geometry in ("staggered-cycle", "frames"):
                            for method in ["rank", "mirror-vc"]:
                                result = recover_direction(points, ranking, method=method)
                                record(
                                    group,
                                    geometry,
                                    method,
                                    result.direction,
                                    gradient,
                                    radius=radius,
                                )
            elif args.experiment == "curvature":
                points = queries("staggered-cycle")
                for gamma in settings["gammas"]:
                    ranking = order(points, gradient, gamma * hessian)
                    centers(group, points, ranking, gradient, "staggered-cycle", gamma=gamma)
                    for method in ["linear", "rank"]:
                        result = recover_direction(
                            points, ranking, method=method, linear_C=settings["linear_C"]
                        )
                        record(
                            group,
                            "staggered-cycle",
                            method,
                            result.direction,
                            gradient,
                            gamma=gamma,
                        )
                    frames = queries("frames")
                    centers(
                        group,
                        frames,
                        order(frames, gradient, gamma * hessian),
                        gradient,
                        "frames",
                        gamma=gamma,
                    )
            elif args.experiment == "random-mirrors":
                lines = load(group["random_mirror_file"])["lines"]
                for radius in settings["radii"]:
                    for geometry in ["random-mirrors", "staggered-cycle"]:
                        points = (
                            radius * torch.cat((lines, -lines), 1)
                            if geometry == "random-mirrors"
                            else queries(geometry, radius)
                        )
                        centers(
                            group,
                            points,
                            order(points, gradient, hessian),
                            gradient,
                            geometry,
                            radius=radius,
                        )
            elif args.experiment == "linear-calibration":
                points = data["points"]
                for gamma in settings["gammas"]:
                    ranking = order(points, gradient, gamma * hessian)
                    for C in settings["C_grid"]:
                        result = recover_direction(points, ranking, method="linear", linear_C=C)
                        record(
                            group,
                            "staggered-cycle",
                            "linear",
                            result.direction,
                            gradient,
                            gamma=gamma,
                            C=C,
                        )
            elif args.experiment == "mismatch":
                points = data["points"]
                exact = torch.einsum("bnd,bd->bn", points, gradient) + 0.5 * torch.einsum(
                    "bni,bij,bnj->bn", points, hessian, points
                )
                rms = (exact - exact.mean(-1, keepdim=True)).square().mean(-1).sqrt()
                cubic = (
                    torch.einsum("bnd,bkd->bnk", points, data["cubic_directions"]).pow(3)
                    * data["cubic_weights"][:, None]
                ).sum(-1)
                cubic -= cubic.mean(-1, keepdim=True)
                cubic_rms = cubic.square().mean(-1).sqrt()
                for strength in settings["strengths"]:
                    scores = torch.asinh(exact + (strength * rms / cubic_rms)[:, None] * cubic) + 2
                    ranking = scores.argsort(dim=-1, stable=True)
                    result = recover_direction(
                        points, ranking, method="hybrid-vc", repair_C=settings["repair_C"]
                    )
                    hard = result.diagnostics["hard_success"].cpu().tolist()
                    record(
                        group,
                        "staggered-cycle",
                        "hybrid-vc",
                        result.direction,
                        gradient,
                        strength=strength,
                        hard=hard,
                    )
                    for method in ["rank", "soft-quadratic"]:
                        result = recover_direction(
                            points, ranking, method=method, repair_C=settings["repair_C"]
                        )
                        record(
                            group,
                            "staggered-cycle",
                            method,
                            result.direction,
                            gradient,
                            strength=strength,
                            hard=hard,
                        )

        if args.experiment == "ambiguity":
            data = load("data/inputs/ambiguity.npz")
            if args.dimension not in (None, 8):
                raise ValueError("ambiguity uses dimension 8")
            batches = list(range(0, len(data["indices"]), 8))[args.shard_index :: args.shards]
            if args.max_groups:
                batches = batches[: args.max_groups]
            for lo in batches:
                try:
                    sl = slice(lo, lo + 8)
                    g1 = data["gradient_one"][sl]
                    g2 = data["gradient_two"][sl]
                    h = data["hessian"][sl]
                    bases = torch.eye(8, device=device, dtype=torch.float64).expand(
                        len(g1), 3, -1, -1
                    )
                    truth_angle = decoders.angular_mse(g1, g2).sqrt()
                    for geometry in ["staggered-cycle", "staggered-two"]:
                        p = make_queries(
                            8, batch_size=len(g1), bases=bases, geometry=geometry, device=device
                        )
                        ranks = [order(p, g, h) for g in [g1, g2]]
                        first = recover_direction(p, ranks[0])
                        second = (
                            first
                            if geometry == "staggered-two"
                            else recover_direction(p, ranks[1])
                        )
                        losses = torch.stack(
                            [
                                decoders.angular_mse(first.direction, g1),
                                decoders.angular_mse(second.direction, g2),
                            ],
                            1,
                        )
                        for i in range(len(g1)):
                            emit(
                                dict(
                                    kind="ambiguity",
                                    pair=int(data["indices"][lo + i]),
                                    geometry=geometry,
                                    same_order=bool(torch.equal(ranks[0][i], ranks[1][i])),
                                    squared_angle_radians=losses[i].cpu().tolist(),
                                    disconnected_bound_radians=float(truth_angle[i] / 2),
                                )
                            )
                except (ArithmeticError, RuntimeError, AssertionError) as error:
                    failures += 1
                    emit(dict(kind="failure", pair_batch=lo, error=str(error)))
        else:
            groups = [g for g in manifest["groups"] if g["cohort"] == settings["cohort"]]
            if "replicas" in settings:
                groups = [g for g in groups if g["replica"] in settings["replicas"]]
            if args.dimension:
                groups = [g for g in groups if g["dimension"] == args.dimension]
            groups = groups[args.shard_index :: args.shards]
            if args.max_groups:
                groups = groups[: args.max_groups]
            if not groups:
                raise ValueError("no groups match this selection")
            for group in groups:
                print(f"{args.experiment}: {group['id']}", flush=True)
                try:
                    run(group)
                except (ArithmeticError, RuntimeError, AssertionError) as error:
                    failures += 1
                    emit(dict(kind="failure", group=group["id"], error=str(error)))
        emit(dict(kind="completion", failed_groups=failures))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
