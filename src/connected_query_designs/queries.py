"""Staggered coordinate-circle queries with orthogonal bridge directions."""

import math
import torch
from .graphs import cycle_edges


@torch.no_grad()
def staggered_bridges(basis, *, disconnected=False, phase=math.pi / 8):
    b, d, _ = basis.shape
    assert d >= 8 and d % 2 == 0
    edges = torch.tensor(cycle_edges(d, disconnected=disconnected), device=basis.device)
    first, second = (edges[: d // 2], edges[d // 2 :])
    theta = phase + torch.arange(4, device=basis.device, dtype=basis.dtype) * math.pi / 4
    local = (
        theta.cos()[None, None, :, None] * basis[:, first[:, 0], None]
        + theta.sin()[None, None, :, None] * basis[:, first[:, 1], None]
    ).flatten(1, 2)
    theta = math.pi / 4 + torch.arange(2, device=basis.device, dtype=basis.dtype) * math.pi / 2
    bridge = (
        theta.cos()[None, None, :, None] * basis[:, second[:, 0], None]
        + theta.sin()[None, None, :, None] * basis[:, second[:, 1], None]
    ).flatten(1, 2)
    z = torch.cat((local, bridge), 1)
    eye = torch.eye(d, device=z.device, dtype=z.dtype)
    tight_error = float((z.transpose(-1, -2) @ z - 3 * eye).abs().max())
    unit_error = float((z.norm(dim=-1) - 1).abs().max())
    assert tight_error < 1e-10 and unit_error < 1e-11 and (z.shape == (b, 3 * d, d))
    cross = local @ bridge.transpose(-1, -2)
    expected_cross = math.cos(math.pi / 8) / math.sqrt(2) if phase == math.pi / 8 else None
    if expected_cross is not None:
        assert abs(float(cross.abs().max()) - expected_cross) < 1e-10
    return (
        z,
        dict(
            unsigned_endpoints=3 * d,
            components=2 if disconnected else 1,
            primary_phase=phase,
            first_matching_lines=4,
            bridge_lines=2,
            expected_feature_dimension=3 * d - 1,
            unit_error=unit_error,
            frame_matrix_error=tight_error,
            cross_maximum_coherence=float(cross.abs().max()),
            support_edges=edges.cpu().tolist(),
            theory="Same connected graph support as the original circle family; phase rotation preserves dense-circle support.",
        ),
    )
