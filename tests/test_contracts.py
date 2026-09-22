"""Independent mathematical identities and regression against frozen paper outputs."""

import csv
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from connected_query_designs import make_queries, recover_direction
from connected_query_designs.api import scores_from_order
from connected_query_designs import decoders
from connected_query_designs.features import complete_mirrored_svd
from connected_query_designs.centers_vc import derivatives, objective

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.cuda


@pytest.mark.parametrize(
    "geometry,expected_rank",
    [
        ("staggered-cycle", 29),
        ("staggered-two", 29),
        ("cycle", 29),
        ("frames", 37),
        ("random-mirrors", 39),
    ],
)
def test_geometry_kernel_and_rank(frozen, device, geometry, expected_rank):
    points = make_queries(
        10, batch_size=8, bases=frozen["bases"], device=device, geometry=geometry
    )
    assert points.shape == (8, 60, 10)
    torch.testing.assert_close(
        points.norm(dim=-1), torch.ones_like(points[:, :, 0]), atol=1e-12, rtol=1e-12
    )
    scores = decoders.quadratic_scores(points, frozen["gradient"], frozen["hessian"])
    order = scores.argsort(dim=-1, stable=True)
    system = complete_mirrored_svd(points, scores_from_order(points, order))
    x = points.gather(1, order[..., None].expand_as(points))
    gram = (1 + x @ x.transpose(-1, -2)).square()
    gram = (
        gram
        - gram.mean(-1, keepdim=True)
        - gram.mean(-2, keepdim=True)
        + gram.mean((-1, -2), keepdim=True)
    )
    features = system["feature"]
    assert features.shape[-1] == expected_rank
    torch.testing.assert_close(features @ features.transpose(-1, -2), gram, atol=2e-12, rtol=2e-12)


def test_rank_sum_and_translation(device):
    rng = torch.Generator(device=device).manual_seed(915)
    points = torch.randn(2, 9, 4, device=device, dtype=torch.float64, generator=rng) + 3
    order = torch.rand(2, 9, device=device, generator=rng).argsort(dim=-1)
    ordered = points.gather(1, order[..., None].expand_as(points))
    i, j = torch.triu_indices(9, 9, 1, device=device)
    pair_sum = (ordered[:, j] - ordered[:, i]).sum(1)
    expected = pair_sum / pair_sum.norm(dim=-1, keepdim=True)
    actual = recover_direction(points, order, method="rank").direction
    translated = recover_direction(points + 5, order, method="rank").direction
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(actual, translated, atol=1e-13, rtol=1e-13)


def test_saved_main_vc_and_hybrid_bypass(frozen, device):
    points = make_queries(10, batch_size=8, bases=frozen["bases"], device=device)
    order = decoders.quadratic_scores(points, frozen["gradient"], frozen["hessian"]).argsort(
        dim=-1
    )
    dtype = torch.get_default_dtype()
    hard = recover_direction(points, order)
    hybrid = recover_direction(points, order, method="hybrid-vc")
    assert torch.get_default_dtype() == dtype
    assert hybrid.diagnostics["hard_success"].all()
    assert torch.equal(hard.direction, hybrid.direction)
    with (ROOT / "data/reference/main.csv").open() as f:
        rows = [
            r
            for r in csv.DictReader(f)
            if r["target_id"].startswith("main/rep0-stream0-regime0-d10/")
            and r["geometry"] == "staggered-cycle"
            and r["decoder"] == "quadratic-vc"
            and float(r["radius"]) == 1.0
        ]
    rows.sort(key=lambda r: int(r["target_id"].split("/")[-1]))
    expected = torch.tensor(
        [float(r["squared_angle_radians"]) for r in rows], device=device, dtype=torch.float64
    )
    torch.testing.assert_close(
        decoders.angular_mse(hard.direction, frozen["gradient"]), expected, atol=1e-10, rtol=1e-7
    )
    reversed_result = recover_direction(points, order.flip(-1))
    torch.testing.assert_close(hard.direction, -reversed_result.direction, atol=2e-8, rtol=2e-8)


def test_order_validation(frozen, device):
    points = make_queries(10, batch_size=8, bases=frozen["bases"], device=device)
    order = torch.arange(60, device=device).expand(8, -1).clone()
    order[:, 0] = order[:, 1]
    with pytest.raises(ValueError, match="permutation"):
        recover_direction(points, order)


def test_vc_derivatives_against_autograd(device):
    rng = torch.Generator(device=device).manual_seed(481)
    a = torch.rand(1, 12, 4, device=device, dtype=torch.float64, generator=rng) + 0.2
    w = torch.ones(1, 4, device=device, dtype=torch.float64, requires_grad=True)
    slack = torch.einsum("bnd,bd->bn", a, w)
    b = a / slack[..., None]
    gram = torch.eye(4, device=device, dtype=w.dtype) + b.transpose(-1, -2) @ b
    loss = (0.5 * w.square().sum(-1) + 0.5 * torch.linalg.slogdet(gram).logabsdet).sum()
    automatic = torch.autograd.grad(loss, w, create_graph=True)[0]
    hessian = torch.stack(
        [torch.autograd.grad(automatic[0, i], w, retain_graph=True)[0][0] for i in range(4)]
    )
    result = derivatives(a, w.detach())
    torch.testing.assert_close(result["gradient"], automatic, atol=1e-12, rtol=1e-12)
    computed = result["design"].transpose(-1, -2) @ result["design"]
    torch.testing.assert_close(computed[0], hessian, atol=1e-12, rtol=1e-12)


def test_known_linear_feasibility_positive_and_negative(device):
    points = torch.tensor([[[-1.0], [0.0], [1.0]]], device=device, dtype=torch.float64)
    yes = decoders.linear_feasibility(points, points.new_tensor([[0.0, 1.0, 2.0]]))
    no = decoders.linear_feasibility(points, points.new_tensor([[0.0, 2.0, 1.0]]))
    assert yes[0]["status"] == "feasible"
    assert no[0]["status"] == "infeasible"


def test_repair_on_saved_cubic_failures(device):
    with np.load(
        ROOT / "data/inputs/robustness/rep0-stream0-regime0-d50.npz", allow_pickle=False
    ) as f:
        data = {k: torch.as_tensor(f[k], device=device) for k in f.files}
    p, g, h = data["points"], data["gradient"], data["hessian"]
    exact = torch.einsum("bnd,bd->bn", p, g) + 0.5 * torch.einsum("bni,bij,bnj->bn", p, h, p)
    rms = (exact - exact.mean(-1, keepdim=True)).square().mean(-1).sqrt()
    cubic = (
        torch.einsum("bnd,bkd->bnk", p, data["cubic_directions"]).pow(3)
        * data["cubic_weights"][:, None]
    ).sum(-1)
    cubic -= cubic.mean(-1, keepdim=True)
    scores = (
        torch.asinh(exact + (0.03 * rms / cubic.square().mean(-1).sqrt())[:, None] * cubic) + 2
    )
    result = recover_direction(p, scores.argsort(dim=-1), method="hybrid-vc")
    with (ROOT / "data/reference/mismatch.csv").open() as f:
        rows = [
            r
            for r in csv.DictReader(f)
            if r["target_id"].startswith("robustness/rep0-stream0-regime0-d50/")
            and r["decoder"] == "hybrid-vc"
            and float(r["strength"]) == 0.03
        ]
    rows.sort(key=lambda r: int(r["target_id"].split("/")[-1]))
    assert len(rows) == 8
    assert result.diagnostics["fallback_used"].any()
    expected = torch.tensor(
        [float(r["squared_angle_radians"]) for r in rows], device=device, dtype=torch.float64
    )
    torch.testing.assert_close(
        decoders.angular_mse(result.direction, g), expected, atol=1e-9, rtol=1e-6
    )
