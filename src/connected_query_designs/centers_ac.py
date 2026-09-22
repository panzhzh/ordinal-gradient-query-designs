"""Batched analytic-center recovery with strict ranking and Newton residual checks."""

import torch


@torch.no_grad()
def spherical_centers(system, initial, max_steps=160):
    phi = system["feature"]
    a = phi.diff(dim=1)
    a = a / a.norm(dim=-1, keepdim=True)
    batch, m, p = a.shape
    eye = torch.eye(p, device=a.device, dtype=a.dtype).expand(batch, -1, -1)
    w = initial / initial.norm(dim=-1, keepdim=True) * m**0.5
    assert bool((torch.einsum("bmi,bi->bm", a, w) > 0).all())
    stopped = torch.zeros(batch, device=a.device, dtype=torch.bool)
    iterations = torch.zeros(batch, device=a.device, dtype=torch.int64)
    for iteration in range(max_steps):
        slack = torch.einsum("bmi,bi->bm", a, w)
        scaled = a / slack[..., None]
        design = torch.cat((eye, scaled), 1)
        residual = torch.cat((w, -torch.ones_like(slack)), 1)
        q, r = torch.linalg.qr(design, mode="reduced")
        projected = (q.transpose(-1, -2) @ residual[..., None]).squeeze(-1)
        step = torch.linalg.solve_triangular(r, projected[..., None], upper=True).squeeze(-1)
        decrement = projected.square().sum(-1)
        grad = w - scaled.sum(1)
        scale = w.abs() + (a.abs() / slack[..., None]).sum(1)
        backward = grad.norm(dim=-1) / (1 + scale.norm(dim=-1))
        newly = (decrement <= 1e-14) & (backward <= 5e-11)
        iterations = torch.where(~stopped, torch.full_like(iterations, iteration + 1), iterations)
        stopped |= newly
        if bool(stopped.all()):
            break
        step = torch.where(stopped[:, None], 0.0, step)
        moved = -torch.einsum("bmi,bi->bm", a, step)
        length = (
            torch.where(moved < 0, -0.99 * slack / moved, torch.inf).min(-1).values.clamp_max(1.0)
        )
        objective = 0.5 * w.square().sum(-1) - slack.log().sum(-1)
        accepted = torch.zeros_like(stopped)
        for _ in range(50):
            relative = length[:, None] * moved / slack
            dw = -length[:, None] * step
            trial = w + dw
            strict = (relative > -1).all(-1) & (torch.einsum("bmi,bi->bm", a, trial) > 0).all(-1)
            change = (
                -torch.log1p(relative.clamp_min(-1 + torch.finfo(w.dtype).eps)).sum(-1)
                + (w * dw).sum(-1)
                + 0.5 * dw.square().sum(-1)
            )
            accepted = strict & (
                change
                <= -0.0001 * length * torch.where(stopped, 0.0, decrement)
                + 32 * torch.finfo(w.dtype).eps * (1 + objective.abs())
            )
            if bool(accepted.all()):
                break
            length = torch.where(accepted, length, length * 0.5)
        if not bool(accepted.all()):
            raise ArithmeticError("Batched ranking-barrier line search failed")
        w = trial
    slack = torch.einsum("bmi,bi->bm", a, w)
    scaled = a / slack[..., None]
    gradient = w - scaled.sum(1)
    scale = w.abs() + (a.abs() / slack[..., None]).sum(1)
    backward = gradient.norm(dim=-1) / (1 + scale.norm(dim=-1))
    q, _ = torch.linalg.qr(torch.cat((eye, scaled), 1), mode="reduced")
    projected = (
        q.transpose(-1, -2) @ torch.cat((w, -torch.ones_like(slack)), 1)[..., None]
    ).squeeze(-1)
    decrement = projected.square().sum(-1)
    allowance = 64 * torch.finfo(w.dtype).eps * torch.einsum("bmi,bi->bm", a.abs(), w.abs())
    strict = (slack > allowance).all(-1)
    raw = (w[:, None] @ system["center_derivative"]).squeeze(1)
    direction = raw / raw.norm(dim=-1, keepdim=True)
    checks = (
        strict & (decrement <= 1e-12) & (backward <= 1e-10) & torch.isfinite(direction).all(-1)
    )
    return dict(
        direction=direction,
        coefficients=w,
        certified=checks,
        strict_feasible=strict,
        newton_decrement_squared=decrement,
        gradient_backward_residual=backward,
        minimum_margin=slack.min(-1).values,
        iterations=iterations,
    )
