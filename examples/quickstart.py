"""Recover an ascent direction from one complete ranking of 6d endpoints."""

import argparse
import torch
from connected_query_designs import make_queries, recover_direction

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()
torch.set_num_threads(1)
torch.cuda.set_device(args.device)
points = make_queries(10, batch_size=2, seed=7, device=args.device)
rng = torch.Generator(device=args.device).manual_seed(19)
gradient = torch.randn(2, 10, dtype=torch.float64, device=args.device, generator=rng)
gradient /= gradient.norm(dim=-1, keepdim=True)
raw = torch.randn(2, 10, 10, dtype=torch.float64, device=args.device, generator=rng)
hessian = (raw + raw.transpose(-1, -2)) / 2
# Synthetic oracle: only its ascending permutation is sent to the decoder.
values = torch.einsum("bnd,bd->bn", points, gradient) + 0.5 * torch.einsum(
    "bni,bij,bnj->bn", points, hessian, points
)
order = values.argsort(dim=-1, stable=True)
result = recover_direction(points, order)
dot = (result.direction * gradient).sum(-1).clamp(-1, 1)
angle = torch.atan2((result.direction - dot[:, None] * gradient).norm(dim=-1), dot)
print("Certified:", result.certified.cpu().tolist())
print("Angular error (degrees):", torch.rad2deg(angle).cpu().tolist())
print("Use -result.direction for a descent step.")
