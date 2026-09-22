"""Public interface: construct queries, then recover from their ascending order."""

from dataclasses import dataclass
import math
from typing import Any

import torch

from . import decoders
from .features import mixed_basis_design
from .orthogonal import positive_diagonal_qr
from .queries import staggered_bridges


@dataclass(frozen=True)
class Recovery:
    """Normalized ascent direction, convergence certificate, and solver details."""

    direction: torch.Tensor
    certified: torch.Tensor
    diagnostics: dict[str, Any]


def _points(points):
    if points.ndim != 3 or not points.is_cuda or points.dtype != torch.float64:
        raise ValueError(
            "points must be a CUDA float64 tensor of shape [batch, endpoints, dimension]"
        )
    if not all(points.shape) or not bool(torch.isfinite(points).all()):
        raise ValueError("points must be nonempty and finite")


def scores_from_order(points: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """Encode a permutation as integer ranks, discarding all score magnitudes."""
    _points(points)
    if (
        order.shape != points.shape[:2]
        or order.dtype != torch.int64
        or order.device != points.device
    ):
        raise ValueError("order must be int64 [batch, endpoints] on the points device")
    expected = torch.arange(points.shape[1], device=points.device).expand_as(order)
    if not torch.equal(order.sort(-1).values, expected):
        raise ValueError("each order row must be a permutation of endpoint indices")
    return torch.zeros_like(order, dtype=points.dtype).scatter_(
        1, order, expected.to(points.dtype)
    )


@torch.no_grad()
def make_queries(
    dimension: int,
    *,
    batch_size: int = 1,
    radius: float = 1.0,
    geometry: str = "staggered-cycle",
    seed: int = 0,
    device: str | torch.device = "cuda",
    bases: torch.Tensor | None = None,
    phase: float = math.pi / 8,
) -> torch.Tensor:
    """Return 6d endpoints in [+z_1, ..., +z_3d, -z_1, ..., -z_3d] order.

    Supported geometries: staggered-cycle, staggered-two, cycle, cycle-two,
    frames, random-mirrors. Supplied bases have shape [batch, >=3, d, d], with
    orthonormal rows. Circle designs use the first basis; Frames uses three.
    """
    if dimension < 8 or dimension % 2 or batch_size < 1:
        raise ValueError("use an even dimension >= 8 and a positive batch size")
    if not math.isfinite(radius) or radius <= 0 or not math.isfinite(phase):
        raise ValueError("radius must be positive and phase finite")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("query construction and recovery require a CUDA device")
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.cuda.device(device):
        if bases is None:
            raw = torch.randn(
                batch_size,
                3,
                dimension,
                dimension,
                device=device,
                dtype=torch.float64,
                generator=generator,
            )
            bases = positive_diagonal_qr(raw)[0]
        if (
            bases.ndim != 4
            or bases.shape[0] != batch_size
            or bases.shape[1] < 3
            or bases.shape[-2:] != (dimension, dimension)
            or bases.dtype != torch.float64
            or bases.device != torch.empty(0, device=device).device
        ):
            raise ValueError("bases must be CUDA float64 [batch, >=3, dimension, dimension]")
        eye = torch.eye(dimension, device=device, dtype=bases.dtype)
        if not torch.allclose(bases @ bases.transpose(-1, -2), eye, atol=1e-10, rtol=1e-10):
            raise ValueError("bases must be orthogonal")
        if geometry in ("staggered-cycle", "staggered-two"):
            lines, _ = staggered_bridges(
                bases[:, 0], disconnected=geometry.endswith("two"), phase=phase
            )
        elif geometry in ("cycle", "cycle-two", "frames"):
            lines, _ = mixed_basis_design(bases, geometry, generator)
        elif geometry == "random-mirrors":
            lines = torch.randn(
                batch_size,
                3 * dimension,
                dimension,
                device=device,
                dtype=torch.float64,
                generator=generator,
            )
            lines /= lines.norm(dim=-1, keepdim=True)
        else:
            raise ValueError(f"unknown geometry: {geometry}")
        return radius * torch.cat((lines, -lines), dim=1)


@torch.no_grad()
def recover_direction(
    points: torch.Tensor,
    order: torch.Tensor,
    *,
    method: str = "quadratic-vc",
    linear_C: float = 1e4,
    repair_C: float = 1e5,
) -> Recovery:
    """Recover the ascent direction using query coordinates and a strict order.

    ``order[:, 0]`` indexes the lowest-valued endpoint. Use the negative of the
    returned direction for descent. Methods: quadratic-ac, quadratic-vc,
    hybrid-vc, soft-quadratic, mirror-ac, mirror-vc, linear, rank.
    Quadratic and mirror methods require exact antipodal pairs in make_queries
    order. An uncertified result raises ArithmeticError; it is never hidden.
    """
    scores = scores_from_order(points, order)
    if method not in ("rank", "linear"):
        n = points.shape[1] // 2
        if points.shape[1] != 2 * n or not torch.equal(points[:, n:], -points[:, :n]):
            raise ValueError("quadratic/mirror decoders require [+z, -z] endpoint order")
    with torch.cuda.device(points.device):
        if method == "rank":
            direction = decoders.rank_direction(points, scores)
            return Recovery(
                direction, torch.ones(len(points), device=points.device, dtype=torch.bool), {}
            )
        if method in ("quadratic-ac", "quadratic-vc"):
            result, info = decoders.quadratic_centers(points, scores)
        elif method in ("mirror-ac", "mirror-vc"):
            result, info = decoders.mirror_centers(points, scores)
        elif method == "hybrid-vc":
            result, info = decoders.robust_quadratic_centers(points, scores, fallback_C=repair_C)
        elif method == "linear":
            result = decoders.linear_soft_direction(points, scores, C=linear_C)
            info = {
                k: v
                for k, v in result.items()
                if k not in ("direction", "coefficients", "certified")
            }
        elif method == "soft-quadratic":
            result = decoders.quadratic_multiscale_soft_direction(
                points, scores, C=repair_C, schedule="dyadic"
            )
            info = {
                k: v
                for k, v in result.items()
                if k not in ("direction", "coefficients", "certified")
            }
        else:
            raise ValueError(f"unknown method: {method}")
        direction, certified = result["direction"], result["certified"]
        if direction.ndim == 3:
            index = 0 if method.endswith("-ac") else 1
            direction, certified = direction[:, index], certified[:, index]
        if not bool(certified.all()):
            raise ArithmeticError(f"{method} did not meet the solver's certification criteria")
        return Recovery(direction, certified, info)
