"""Coordinate-support graphs and budget-matched query grids."""

from __future__ import annotations
import math
import torch


def cycle_edges(d, *, disconnected=False):
    edges = [(i, i + 1) for i in range(0, d, 2)]
    sections = [(0, d)]
    if disconnected:
        split = 2 * (d // 4)
        sections = [(0, split), (split, d)]
    for lo, hi in sections:
        edges += [(i, i + 1) for i in range(lo + 1, hi - 1, 2)]
        edges.append((hi - 1, lo))
    return edges


def graph_grid_design(basis, k, variant, *, generator):
    b, d, other = basis.shape
    if d != other or d < 8 or d % 2 or (k < 3) or (k % 2 == 0):
        raise ValueError("Even d >= 8 and odd K >= 3 required")
    if variant.startswith("chord"):
        z, edge, check = graph_grid_design(basis, k, "cycle", generator=generator)
        count = {"chord1": 1, "chord-eighth": max(1, d // 8), "chord-quarter": d // 4}[variant]
        even = 2 * torch.randperm(d // 2, device=basis.device, generator=generator)
        take = even[: 2 * count].reshape(count, 2)
        for i, j in take.tolist():
            z[:, i] = (basis[:, i] + basis[:, j]) / math.sqrt(2)
            z[:, j] = (basis[:, i] - basis[:, j]) / math.sqrt(2)
        edge = torch.cat((edge, take), 0)
        assert (
            float(
                (z.transpose(-1, -2) @ z - k * torch.eye(d, device=z.device, dtype=z.dtype))
                .abs()
                .max()
            )
            < 1e-10
        )
        return (
            z,
            edge,
            {
                **check,
                "base_cycle_degrees": check["degrees"],
                "degrees": torch.bincount(edge.flatten(), minlength=d).cpu().tolist(),
                "extra_chords": count,
                "replaced_axis_lines": 2 * count,
                "maximum_feature_dimension": 3 * d - 1 + count,
            },
        )
    edges = cycle_edges(d, disconnected=variant.endswith("-two"))
    if variant == "balanced":
        edges = [((i - 1) // 2, i) for i in range(1, d)] + [(d - 2, d - 1)]
    elif variant == "star":
        edges = [(0, i) for i in range(1, d)] + [(1, 2)]
    edge = torch.tensor(edges, dtype=torch.int64, device=basis.device)
    first, second = (basis[:, edge[:, 0]], basis[:, edge[:, 1]])
    if variant.startswith("cubic"):
        existing = {tuple(sorted(e)) for e in edges}
        sections = [(0, d)]
        if variant.endswith("-two"):
            split = 2 * (d // 4)
            sections = [(0, split), (split, d)]
        extra = []
        for lo, hi in sections:
            for _ in range(10000):
                perm = (
                    torch.randperm(hi - lo, device=basis.device, generator=generator) + lo
                ).tolist()
                candidate = list(zip(perm[::2], perm[1::2]))
                if all((tuple(sorted(e)) not in existing for e in candidate)):
                    extra.extend(candidate)
                    break
            else:
                raise RuntimeError("No admissible public third matching")
        edges += extra
        edge = torch.tensor(edges, dtype=torch.int64, device=basis.device)
        parts = []
        for matching in range(3):
            count = k // 3 + (matching < k % 3)
            take = edge[matching * d // 2 : (matching + 1) * d // 2]
            a, c = (basis[:, take[:, 0]], basis[:, take[:, 1]])
            theta = (
                torch.arange(1, count + 1, device=basis.device, dtype=basis.dtype)
                * math.pi
                / (2 * (count + 1))
            )
            theta = theta.expand(b, d // 2, count).clone()
            if "jitter" in variant:
                theta += (
                    (
                        torch.rand(
                            (b, d // 2, 1),
                            device=basis.device,
                            dtype=basis.dtype,
                            generator=generator,
                        )
                        - 0.5
                    )
                    * math.pi
                    / (2 * (count + 1))
                )
            u = theta.cos()[..., None] * a[:, :, None] + theta.sin()[..., None] * c[:, :, None]
            v = theta.sin()[..., None] * a[:, :, None] - theta.cos()[..., None] * c[:, :, None]
            parts.append(torch.stack((u, v), 3).reshape(b, count * d, d))
        z = torch.cat(parts, 1)
    elif variant.startswith("circle-"):
        if "random" in variant:
            phase = (
                torch.rand((b, d, 1), device=basis.device, dtype=basis.dtype, generator=generator)
                * math.pi
                / k
            )
        else:
            phase = basis.new_full((b, d, 1), math.pi / (4 * k))
        theta = phase + torch.arange(k, device=basis.device, dtype=basis.dtype) * math.pi / k
        z = (
            theta.cos()[..., None] * first[:, :, None]
            + theta.sin()[..., None] * second[:, :, None]
        ).reshape(b, k * d, d)
    else:
        j = (k - 1) // 2
        step = math.pi / (2 * (j + 1))
        theta = torch.arange(1, j + 1, device=basis.device, dtype=basis.dtype) * step
        theta = theta.expand(b, d, j).clone()
        if variant == "phase30":
            theta -= step / 3
        elif variant == "jitter":
            theta += (
                torch.rand((b, d, 1), device=basis.device, dtype=basis.dtype, generator=generator)
                - 0.5
            ) * step
        elif variant not in ("cycle", "cycle-two", "balanced", "star"):
            raise ValueError(f"Unknown grid candidate: {variant}")
        a = (
            theta.cos()[..., None] * first[:, :, None]
            + theta.sin()[..., None] * second[:, :, None]
        )
        c = (
            theta.sin()[..., None] * first[:, :, None]
            - theta.cos()[..., None] * second[:, :, None]
        )
        z = torch.cat((basis, torch.stack((a, c), 3).reshape(b, 2 * j * d, d)), 1)
    assert z.shape == (b, k * d, d)
    degree = torch.bincount(edge.flatten(), minlength=d)
    norm_error = float((z.norm(dim=-1) - 1).abs().max())
    tight_error = float(
        (z.transpose(-1, -2) @ z - k * torch.eye(d, device=z.device, dtype=z.dtype)).abs().max()
    )
    assert norm_error < 1e-11
    if variant not in ("balanced", "star"):
        assert tight_error < 1e-10
    return (
        z,
        edge,
        {
            "unit_error": norm_error,
            "tight_error": tight_error,
            "degrees": degree.cpu().tolist(),
            "unsigned_endpoints": k * d,
        },
    )


@torch.no_grad()
def sparse_graph_polynomial_system(points, scores, basis, edges):
    """Exact feature factorization for arbitrary unit two-coordinate queries.

    An SVD of the small complete feature matrix handles non-tight grids, phase
    changes, and all diagonal/cross couplings. No design-specific Gram formula
    or statistical prior is assumed. The independent dense kernel audit checks
    the RKHS norm and the center derivative used by the unchanged RankSVM.
    """
    b, n, d = points.shape
    values, order = scores.sort(dim=-1, stable=True)
    if not bool((values.diff(dim=-1) > 0).all()):
        raise ValueError("Strict ranks required")
    x = points.gather(1, order[..., None].expand_as(points))
    local = x @ basis.transpose(-1, -2)
    t = x.new_zeros((d - 1, d))
    for i in range(1, d):
        t[i - 1, :i] = 1 / math.sqrt(i * (i + 1))
        t[i - 1, i] = -i / math.sqrt(i * (i + 1))
    explicit = torch.cat(
        (
            math.sqrt(2) * local,
            local.square() @ t.T,
            math.sqrt(2) * local[..., edges[:, 0]] * local[..., edges[:, 1]],
        ),
        -1,
    )
    explicit -= explicit.mean(1, keepdim=True)
    u, s, vh = torch.linalg.svd(explicit, full_matrices=False)
    cutoff = 64 * torch.finfo(x.dtype).eps * max(n, explicit.shape[-1]) * s[:, :1]
    retained = s > cutoff
    ranks = retained.sum(-1)
    r = int(ranks.max())
    s, u, vh, retained = (s[:, :r], u[:, :, :r], vh[:, :r], retained[:, :r])
    root = torch.where(retained, s, 0.0)
    inverse = torch.where(retained, s.clamp_min(torch.finfo(x.dtype).tiny).reciprocal(), 0.0)
    feature = u * root[:, None]
    derivative = math.sqrt(2) * vh[..., :d] @ basis
    gram = (1 + x @ x.transpose(-1, -2)).square()
    centered = (
        gram
        - gram.mean(-1, keepdim=True)
        - gram.mean(-2, keepdim=True)
        + gram.mean((-1, -2), keepdim=True)
    )
    error = (feature @ feature.transpose(-1, -2) - centered).abs().amax((-1, -2))
    assert float((error / (1 + centered.abs().amax((-1, -2)))).max()) < 1e-11
    return {
        "points": x,
        "order": order,
        "feature": feature,
        "center_derivative": derivative,
        "eigenvectors": u,
        "inverse_roots": inverse,
        "kernel_eigenvalues": root.square(),
        "gram": gram,
        "centered_gram": centered,
        "rank": ranks,
        "kernel_reconstruction_error": error,
        "kind": "poly2",
        "gamma": None,
    }
