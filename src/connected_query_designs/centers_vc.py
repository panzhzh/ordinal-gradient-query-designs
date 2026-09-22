"""Batched volumetric-center recovery in the physical polynomial coefficient metric."""

import torch


def derivatives(a, w):
    batch, m, p = a.shape
    eye = torch.eye(p, device=a.device, dtype=a.dtype).expand(batch, -1, -1)
    slack = torch.einsum("bmi,bi->bm", a, w)
    b = a / slack[..., None]
    q, r = torch.linalg.qr(torch.cat((eye, b), 1), mode="reduced")
    qb = q[:, p:]
    projection = qb @ qb.transpose(-1, -2)
    leverage = projection.diagonal(dim1=-2, dim2=-1)
    k = 3 * torch.diag_embed(leverage) - 2 * projection.square()
    factor = torch.linalg.cholesky(k)
    design = torch.cat((eye, factor.transpose(-1, -2) @ b), 1)
    qh, rh = torch.linalg.qr(design, mode="reduced")
    gradient = w - torch.einsum("bmi,bm->bi", b, leverage)
    middle = torch.linalg.solve_triangular(rh.transpose(-1, -2), gradient[..., None], upper=False)
    step = torch.linalg.solve_triangular(rh, middle, upper=True).squeeze(-1)
    decrement = middle.square().sum((-1, -2))
    objective = 0.5 * w.square().sum(-1) + r.diagonal(dim1=-2, dim2=-1).abs().log().sum(-1)
    scale = w.abs() + torch.einsum("bmi,bm->bi", b.abs(), leverage)
    backward = gradient.norm(dim=-1) / (1 + scale.norm(dim=-1))
    return dict(
        objective=objective,
        gradient=gradient,
        step=step,
        decrement=decrement,
        backward=backward,
        leverage=leverage,
        slack=slack,
        design=design,
    )


def objective(a, w):
    p = a.shape[-1]
    slack = torch.einsum("bmi,bi->bm", a, w)
    eye = torch.eye(p, device=a.device, dtype=a.dtype).expand(len(a), -1, -1)
    _, r = torch.linalg.qr(torch.cat((eye, a / slack[..., None]), 1), mode="reduced")
    return 0.5 * w.square().sum(-1) + r.diagonal(dim1=-2, dim2=-1).abs().log().sum(-1)


@torch.no_grad()
def volumetric_centers(system, initial, max_steps=120):
    a = system["feature"].diff(dim=1)
    a = a / a.norm(dim=-1, keepdim=True)
    w = initial / initial.norm(dim=-1, keepdim=True) * a.shape[-1] ** 0.5
    assert bool((torch.einsum("bmi,bi->bm", a, w) > 0).all())
    stopped = torch.zeros(len(a), device=a.device, dtype=torch.bool)
    iterations = torch.zeros(len(a), device=a.device, dtype=torch.long)
    for iteration in range(max_steps):
        info = derivatives(a, w)
        iterations = torch.where(stopped, iterations, iteration + 1)
        stopped |= (info["decrement"] <= 1e-14) & (info["backward"] <= 5e-11)
        if bool(stopped.all()):
            break
        step = torch.where(stopped[:, None], 0.0, info["step"])
        moved = -torch.einsum("bmi,bi->bm", a, step)
        length = (
            torch.where(moved < 0, -0.99 * info["slack"] / moved, torch.inf)
            .amin(-1)
            .clamp_max(1.0)
        )
        for _ in range(50):
            trial = w - length[:, None] * step
            strict = (torch.einsum("bmi,bi->bm", a, trial) > 0).all(-1)
            value = objective(a, trial)
            accepted = strict & (
                value
                <= info["objective"]
                - 0.0001 * length * torch.where(stopped, 0.0, info["decrement"])
                + 64 * torch.finfo(w.dtype).eps * (1 + info["objective"].abs())
            )
            if bool(accepted.all()):
                break
            length = torch.where(accepted, length, length * 0.5)
        if not bool(accepted.all()):
            raise ArithmeticError("Volumetric-center line search failed")
        w = trial
    info = derivatives(a, w)
    allowance = 64 * torch.finfo(w.dtype).eps * torch.einsum("bmi,bi->bm", a.abs(), w.abs())
    strict = (info["slack"] > allowance).all(-1)
    raw = (w[:, None] @ system["center_derivative"]).squeeze(1)
    direction = raw / raw.norm(dim=-1, keepdim=True)
    certified = (
        strict
        & (info["decrement"] <= 1e-12)
        & (info["backward"] <= 1e-10)
        & torch.isfinite(direction).all(-1)
    )
    return dict(
        direction=direction,
        coefficients=w,
        certified=certified,
        strict_feasible=strict,
        newton_decrement_squared=info["decrement"],
        gradient_backward_residual=info["backward"],
        minimum_margin=info["slack"].amin(-1),
        iterations=iterations,
        minimum_leverage=info["leverage"].amin(-1),
        maximum_leverage=info["leverage"].amax(-1),
    )
