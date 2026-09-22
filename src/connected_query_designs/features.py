"""Mirrored frame controls and an exact SVD factorization of the physical quadratic kernel."""

from __future__ import annotations
import math
import torch
from .graphs import graph_grid_design


@torch.no_grad()
def mixed_basis_design(bases, arm, generator):
    batch, _, d, _ = bases.shape
    if arm == "frames":
        z = bases[:, :3].reshape(batch, 3 * d, d)
    else:
        z, _, _ = graph_grid_design(
            bases[:, 0], 3, "cycle-two" if arm == "cycle-two" else "cycle", generator=generator
        )
        if arm.startswith("mix-"):
            fraction = {"mix-quarter": 0.25, "mix-half": 0.5, "mix-full": 1.0}[arm]
            count = d if fraction == 1 else max(2, 2 * int(d * fraction / 2))
            indices = torch.randperm(d, device=bases.device, generator=generator)[:count]
            rotation = torch.randn(
                (batch, count, count), dtype=bases.dtype, device=bases.device, generator=generator
            )
            q, r = torch.linalg.qr(rotation)
            q *= torch.diagonal(r, dim1=-2, dim2=-1).sign()[:, None]
            z[:, 2 * d + indices] = q @ z[:, 2 * d + indices]
        elif arm not in ("cycle", "cycle-two"):
            raise ValueError(arm)
    eye = torch.eye(d, device=z.device, dtype=z.dtype)
    norms = float((z.norm(dim=-1) - 1).abs().max())
    tight = float((z.transpose(-1, -2) @ z - 3 * eye).abs().max())
    assert norms < 1e-11 and tight < 1e-10
    return (z, dict(unsigned_endpoints=3 * d, unit_error=norms, tight_error=tight))


@torch.no_grad()
def complete_mirrored_svd(points, scores):
    """Complete centered polynomial kernel, factorized by explicit SVD.

    SVD of the even feature matrix avoids determining a small positive rank
    from eigenvalues of its squared, rounded Gram matrix. No feature truncation
    other than the explicit floating-point rank threshold is used.
    """
    batch, n, d = points.shape
    assert points.is_cuda and points.dtype == torch.float64 and (n % 2 == 0)
    m = n // 2
    z = points[:, :m]
    assert torch.equal(points[:, m:], -z)
    values, order = scores.sort(dim=-1, stable=True)
    assert bool((values.diff(dim=-1) > 0).all())
    ii, jj = torch.triu_indices(d, d, 1, device=z.device)
    even_raw = torch.cat((z.square(), math.sqrt(2) * z[..., ii] * z[..., jj]), -1)
    even_raw -= even_raw.mean(1, keepdim=True)
    ue, se, ve = torch.linalg.svd(even_raw, full_matrices=False)
    threshold = 64 * torch.finfo(z.dtype).eps * max(m, even_raw.shape[-1]) * se[:, :1]
    keep = se > threshold
    erank = keep.sum(-1)
    count = int(erank.max())
    se = torch.where(keep[:, :count], se[:, :count], 0.0)
    even = ue[:, :, :count] * se[:, None]
    _, so, vo = torch.linalg.svd(z, full_matrices=False)
    assert bool((so[:, -1] > 1e-10 * so[:, 0]).all())
    odd = math.sqrt(2) * points @ vo.transpose(-1, -2)
    phi = torch.cat((odd, torch.cat((even, even), 1)), -1)
    phi = phi.gather(1, order[..., None].expand(batch, n, d + count))
    spectrum = torch.cat((4 * so.square(), 2 * se.square()), -1)
    assert bool((spectrum > 0).all()), "Batch has unequal feature ranks; split before fitting"
    inverse = spectrum.rsqrt()
    derivative = z.new_zeros((batch, d + count, d))
    derivative[:, :d] = math.sqrt(2) * vo
    sorted_points = points.gather(1, order[..., None].expand_as(points))
    gram = (1 + sorted_points @ sorted_points.transpose(-1, -2)).square()
    centered = (
        gram
        - gram.mean(-1, keepdim=True)
        - gram.mean(-2, keepdim=True)
        + gram.mean((-1, -2), keepdim=True)
    )
    error = (phi @ phi.transpose(-1, -2) - centered).abs().amax((-1, -2))
    assert float((error / (1 + centered.abs().amax((-1, -2)))).max()) < 1e-11
    return dict(
        points=sorted_points,
        order=order,
        feature=phi,
        center_derivative=derivative,
        kernel_eigenvalues=spectrum,
        inverse_roots=inverse,
        eigenvectors=phi * inverse[:, None],
        rank=d + erank,
        gram=gram,
        centered_gram=centered,
        kernel_reconstruction_error=error,
        even_axes=ve[:, :count],
        linear_axes=vo,
        kind="poly2",
        gamma=None,
    )
