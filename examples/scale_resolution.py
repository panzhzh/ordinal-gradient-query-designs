"""Expose the four connected ranking states of the paper's fixed scale family."""

import argparse
import json
import torch
from connected_query_designs import make_queries, recover_direction
from connected_query_designs.decoders import angular_mse, quadratic_scores

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()
torch.set_num_threads(1)
torch.cuda.set_device(args.device)
a = torch.tensor([1.031, 0.913, 1.079, 1.001], device=args.device, dtype=torch.float64)
b = torch.tensor([1.503, 1.437, 1.619, 1.491], device=args.device, dtype=torch.float64)
breaks = [1.0, float(b[3] / a[0]), float((b[0] + b[3]) / (a[3] + a[0])), float(b[0] / a[3]), 2.0]
h = torch.diag(a.new_tensor([10.0] * 4 + [50.0] * 4))[None]
bases = torch.eye(8, device=args.device, dtype=torch.float64).expand(1, 3, -1, -1)
report = {"breakpoints": breaks[1:-1], "geometries": {}}
for geometry in ["staggered-cycle", "staggered-two"]:
    points = make_queries(8, bases=bases, geometry=geometry, device=args.device)
    states = []
    seen = set()
    for left, right in zip(breaks[:-1], breaks[1:]):
        c = (left + right) / 2
        truth = torch.cat((c * a, b))[None]
        order = quadratic_scores(points, truth, h).argsort(dim=-1, stable=True)
        seen.add(tuple(order[0].cpu().tolist()))
        result = recover_direction(points, order)
        states.append(
            dict(
                interval=[left, right],
                representative=c,
                direction=result.direction[0].cpu().tolist(),
                error_degrees=float(torch.rad2deg(angular_mse(result.direction, truth).sqrt())[0]),
            )
        )
    report["geometries"][geometry] = {"distinct_rankings": len(seen), "intervals": states}
assert report["geometries"]["staggered-cycle"]["distinct_rankings"] == 4
assert report["geometries"]["staggered-two"]["distinct_rankings"] == 1
print(json.dumps(report, indent=2))
