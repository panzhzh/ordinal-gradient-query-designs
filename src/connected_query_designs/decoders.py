"""Quadratic, linear, mirror-sign and rank-weighted decoders with ordinal-only inputs."""

import math
import time
import torch
from .centers_ac import spherical_centers
from .centers_vc import volumetric_centers
from .feasibility import rank_feasible_start
from .features import complete_mirrored_svd


def timed(function, *args, **kwargs):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = function(*args, **kwargs)
    torch.cuda.synchronize()
    return (value, time.perf_counter() - start)


def ordered_points(points, scores):
    assert points.is_cuda and points.dtype == torch.float64
    values, order = scores.sort(dim=-1, stable=True)
    if not bool((values.diff(dim=-1) > 0).all()):
        raise ArithmeticError("The observed ranking contains ties")
    return points.gather(1, order[..., None].expand_as(points))


def rank_direction(points, scores):
    """ZO-RankSGD (5), (14)--(16), k=m, up to a positive scalar."""
    ordered = ordered_points(points, scores)
    n = points.shape[1]
    weights = 2 * torch.arange(n, device=points.device, dtype=points.dtype) - n + 1
    raw = (ordered * weights[None, :, None]).sum(1)
    norm = raw.norm(dim=-1, keepdim=True)
    if not bool(torch.isfinite(raw).all() and (norm > 0).all()):
        raise ArithmeticError("Degenerate rank direction")
    return raw / norm


def mirror_constraints(points, scores):
    n = points.shape[1] // 2
    assert points.shape[1] == 2 * n and torch.equal(points[:, n:], -points[:, :n])
    sign = (scores[:, :n] - scores[:, n:]).sign()
    if not bool((sign != 0).all()):
        raise ArithmeticError("A mirror comparison is tied")
    z = points[:, :n]
    return sign[..., None] * z / z.norm(dim=-1, keepdim=True)


def linear_system(constraints):
    """Encode arbitrary rows using consecutive feature differences for AC/VC."""
    batch, _, d = constraints.shape
    phi = torch.cat((torch.zeros_like(constraints[:, :1]), constraints.cumsum(1)), 1)
    assert torch.allclose(phi.diff(dim=1), constraints, atol=2e-14, rtol=2e-14)
    return dict(
        feature=phi,
        center_derivative=torch.eye(d, device=phi.device, dtype=phi.dtype).expand(batch, -1, -1),
    )


def initialize(system):
    phi = system["feature"]
    a = phi.diff(dim=1)
    a = a / a.norm(dim=-1, keepdim=True)
    if "kernel_eigenvalues" in system:
        ranks = torch.linspace(-1, 1, phi.shape[1], device=phi.device, dtype=phi.dtype).expand(
            len(phi), -1
        )
        initial = (phi.transpose(-1, -2) @ ranks[..., None]).squeeze(-1) / system[
            "kernel_eigenvalues"
        ]
    else:
        initial = a.sum(1)
    phase_one = []
    for i in range(len(a)):
        margin = a[i] @ initial[i]
        allowance = 256 * torch.finfo(a.dtype).eps * (a[i].abs() @ initial[i].abs())
        if not bool((margin > allowance).all()):
            result = rank_feasible_start(a[i])
            initial[i] = result["direction"]
            phase_one.append(dict(target=i, history=result["history"]))
    return (initial, phase_one)


def solve_system(system):
    (initial, history), init_seconds = timed(initialize, system)
    ac, ac_seconds = timed(spherical_centers, system, initial)
    vc, vc_seconds = timed(volumetric_centers, system, ac["coefficients"])
    result = dict(
        direction=torch.stack((ac["direction"], vc["direction"]), 1),
        certified=torch.stack((ac["certified"], vc["certified"]), 1),
        iterations=torch.stack((ac["iterations"], vc["iterations"]), 1),
        coefficients=torch.stack((ac["coefficients"], vc["coefficients"]), 1),
    )
    audit = [
        {k: v.cpu().tolist() for k, v in method.items() if k not in ("direction", "coefficients")}
        for method in (ac, vc)
    ]
    return (
        result,
        dict(
            initialization=init_seconds,
            AC=ac_seconds,
            VC_extra=vc_seconds,
            AC_online=init_seconds + ac_seconds,
            VC_online=init_seconds + ac_seconds + vc_seconds,
        ),
        audit,
        history,
    )


def quadratic_centers(points, scores):
    system, feature_seconds = timed(complete_mirrored_svd, points, scores)
    result, timing, audit, history = solve_system(system)
    timing["features_SVD"] = feature_seconds
    timing["AC_online"] += feature_seconds
    timing["VC_online"] += feature_seconds
    spectrum = system["kernel_eigenvalues"]
    return (
        result,
        dict(
            timing=timing,
            audit=audit,
            phase_one=history,
            feature_dimension=system["feature"].shape[-1],
            gram_condition=(spectrum.amax(-1) / spectrum.amin(-1)).cpu().tolist(),
        ),
    )


def mirror_centers(points, scores):

    def construct():
        constraints = mirror_constraints(points, scores)
        return (constraints, linear_system(constraints))

    (constraints, system), construction = timed(construct)
    result, timing, audit, history = solve_system(system)
    for j in range(2):
        slack = (constraints @ result["direction"][:, j, :, None]).squeeze(-1)
        result["certified"][:, j] &= (slack > 128 * torch.finfo(slack.dtype).eps).all(-1)
    timing["features_SVD"] = construction
    timing["AC_online"] += construction
    timing["VC_online"] += construction
    return (
        result,
        dict(timing=timing, audit=audit, phase_one=history, feature_dimension=points.shape[-1]),
    )


def known_linear_centers(points, scores):
    """Oracle knowing H=0; no ground-truth gradient enters fitting."""
    phi, construction = timed(ordered_points, points, scores)
    d = points.shape[-1]
    system = dict(
        feature=phi,
        center_derivative=torch.eye(d, device=points.device, dtype=points.dtype).expand(
            len(points), -1, -1
        ),
    )
    result, timing, audit, history = solve_system(system)
    timing["features_SVD"] = construction
    timing["AC_online"] += construction
    timing["VC_online"] += construction
    return (result, dict(timing=timing, audit=audit, phase_one=history, feature_dimension=d))


def linear_feasibility(points, scores):
    """Check strict linear representability; distinguish infeasibility from failure."""
    constraints = ordered_points(points, scores).diff(dim=1)
    constraints /= constraints.norm(dim=-1, keepdim=True)
    records = []
    for a in constraints:
        try:
            result = rank_feasible_start(a)
            records.append(dict(status="feasible", detail=result["history"]))
        except ArithmeticError as error:
            detail = str(error)
            status = (
                "infeasible"
                if detail == "Convex phase I failed: PrimalInfeasible"
                else "numerical_failure"
            )
            records.append(dict(status=status, detail=detail))
    return records


def linear_soft_direction(points, scores, C=1.0, tolerance=1e-10, maximum_iterations=100):
    """Full-ranking linear squared-hinge RankSVM with a frozen penalty.

    Consecutive differences encode the complete strict order.  Row
    normalization gives each ranking relation the same margin convention.
    The objective is convex and is solved by an active-set Newton method with
    per-instance backtracking.  This is a decoder control, not a feasibility
    oracle: it always returns a direction when the optimum is nonzero.
    """
    if not (C > 0 and tolerance > 0 and (maximum_iterations > 0)):
        raise ValueError("RankSVM constants must be positive")
    constraints = ordered_points(points, scores).diff(dim=1)
    constraints /= constraints.norm(dim=-1, keepdim=True)
    batch, _, dimension = constraints.shape
    coefficient = constraints.sum(1)
    coefficient /= coefficient.norm(dim=-1, keepdim=True).clamp_min(
        torch.finfo(coefficient.dtype).tiny
    )
    identity = torch.eye(dimension, device=coefficient.device, dtype=coefficient.dtype).expand(
        batch, -1, -1
    )
    iterations = torch.zeros(batch, device=coefficient.device, dtype=torch.int64)
    converged = torch.zeros(batch, device=coefficient.device, dtype=torch.bool)

    def derivatives(value):
        margin = torch.einsum("bnd,bd->bn", constraints, value)
        residual = (1 - margin).clamp_min(0)
        objective = 0.5 * value.square().sum(-1) + 0.5 * C * residual.square().sum(-1)
        gradient = value - C * torch.einsum("bnd,bn->bd", constraints, residual)
        active = (residual > 0).to(value.dtype)
        hessian = identity + C * torch.einsum("bni,bn,bnj->bij", constraints, active, constraints)
        return (objective, gradient, hessian)

    for iteration in range(maximum_iterations):
        objective, gradient, hessian = derivatives(coefficient)
        residual_norm = gradient.norm(dim=-1)
        threshold = tolerance * (1 + coefficient.norm(dim=-1))
        just_converged = residual_norm <= threshold
        converged |= just_converged
        if bool(converged.all()):
            break
        step = torch.linalg.solve(hessian, gradient[..., None]).squeeze(-1)
        descent = (gradient * step).sum(-1).clamp_min(0)
        alpha = torch.ones(batch, device=coefficient.device, dtype=coefficient.dtype)
        accepted = converged.clone()
        candidate = coefficient.clone()
        for _ in range(60):
            trial = coefficient - alpha[:, None] * step
            trial_objective = derivatives(trial)[0]
            sufficient = trial_objective <= objective - 0.0001 * alpha * descent
            new = sufficient & ~accepted
            candidate[new] = trial[new]
            accepted |= sufficient
            if bool(accepted.all()):
                break
            alpha[~accepted] *= 0.5
        if not bool(accepted.all()):
            raise ArithmeticError("Linear soft decoder line search failed")
        coefficient = candidate
        iterations[~converged] = iteration + 1
    objective, gradient, _ = derivatives(coefficient)
    gradient_residual = gradient.norm(dim=-1)
    certified = torch.isfinite(coefficient).all(-1) & (
        gradient_residual <= 100 * tolerance * (1 + coefficient.norm(dim=-1))
    )
    norm = coefficient.norm(dim=-1, keepdim=True)
    if not bool(torch.isfinite(norm).all() and (norm > 0).all()):
        raise ArithmeticError("Linear soft decoder returned a zero or nonfinite optimum")
    return dict(
        direction=coefficient / norm,
        coefficients=coefficient,
        certified=certified,
        iterations=iterations,
        objective=objective,
        gradient_residual=gradient_residual,
        penalty_C=coefficient.new_full((batch,), C),
    )


def quadratic_soft_direction(points, scores, C=1.0, tolerance=1e-10, maximum_iterations=100):
    """Soft squared-hinge ranking fit in the complete quadratic feature space.

    This is an explicit fallback for rankings outside the strict quadratic
    cone.  Exact feasible inputs continue to use the unchanged hard AC/VC
    decoder.  The physical SVD map and its coefficient metric are identical
    to ``quadratic_centers``.
    """
    if not (C > 0 and tolerance > 0 and (maximum_iterations > 0)):
        raise ValueError("Soft quadratic constants must be positive")
    system = complete_mirrored_svd(points, scores)
    constraints = system["feature"].diff(dim=1)
    constraints /= constraints.norm(dim=-1, keepdim=True)
    batch, _, dimension = constraints.shape
    coefficient = constraints.sum(1)
    coefficient /= coefficient.norm(dim=-1, keepdim=True).clamp_min(
        torch.finfo(coefficient.dtype).tiny
    )
    identity = torch.eye(dimension, device=coefficient.device, dtype=coefficient.dtype).expand(
        batch, -1, -1
    )
    iterations = torch.zeros(batch, device=coefficient.device, dtype=torch.int64)
    converged = torch.zeros(batch, device=coefficient.device, dtype=torch.bool)

    def derivatives(value):
        margin = torch.einsum("bnd,bd->bn", constraints, value)
        residual = (1 - margin).clamp_min(0)
        objective = 0.5 * value.square().sum(-1) + 0.5 * C * residual.square().sum(-1)
        gradient = value - C * torch.einsum("bnd,bn->bd", constraints, residual)
        active = (residual > 0).to(value.dtype)
        hessian = identity + C * torch.einsum("bni,bn,bnj->bij", constraints, active, constraints)
        return (objective, gradient, hessian)

    for iteration in range(maximum_iterations):
        objective, gradient, hessian = derivatives(coefficient)
        residual_norm = gradient.norm(dim=-1)
        threshold = tolerance * (1 + coefficient.norm(dim=-1))
        converged |= residual_norm <= threshold
        if bool(converged.all()):
            break
        step = torch.linalg.solve(hessian, gradient[..., None]).squeeze(-1)
        descent = (gradient * step).sum(-1).clamp_min(0)
        alpha = torch.ones(batch, device=coefficient.device, dtype=coefficient.dtype)
        accepted = converged.clone()
        candidate = coefficient.clone()
        for _ in range(60):
            trial = coefficient - alpha[:, None] * step
            trial_objective = derivatives(trial)[0]
            sufficient = trial_objective <= objective - 0.0001 * alpha * descent
            new = sufficient & ~accepted
            candidate[new] = trial[new]
            accepted |= sufficient
            if bool(accepted.all()):
                break
            alpha[~accepted] *= 0.5
        if not bool(accepted.all()):
            raise ArithmeticError("Quadratic soft decoder line search failed")
        coefficient = candidate
        iterations[~converged] = iteration + 1
    objective, gradient, _ = derivatives(coefficient)
    gradient_residual = gradient.norm(dim=-1)
    raw = torch.einsum("bp,bpd->bd", coefficient, system["center_derivative"])
    norm = raw.norm(dim=-1, keepdim=True)
    certified = (
        torch.isfinite(coefficient).all(-1)
        & torch.isfinite(raw).all(-1)
        & (norm[:, 0] > 0)
        & (gradient_residual <= 100 * tolerance * (1 + coefficient.norm(dim=-1)))
    )
    if not bool(torch.isfinite(norm).all() and (norm > 0).all()):
        raise ArithmeticError("Quadratic soft decoder returned a zero or nonfinite gradient")
    return dict(
        direction=raw / norm,
        coefficients=coefficient,
        certified=certified,
        iterations=iterations,
        objective=objective,
        gradient_residual=gradient_residual,
        penalty_C=coefficient.new_full((batch,), C),
        feature_dimension=system["feature"].new_full(
            (batch,), system["feature"].shape[-1], dtype=torch.int64
        ),
    )


def quadratic_multiscale_soft_direction(
    points, scores, C=1000.0, schedule="dyadic", tolerance=1e-09, maximum_iterations=100
):
    """Fit and project a soft complete ranking to a quadratic-consistent order.

    The adjacent schedule uses the transitive reduction of the observed order.
    The dyadic schedule additionally compares ranks separated by powers of two,
    giving a scalable O(N log N) surrogate for all pairwise comparisons.  The
    loss is averaged across rows, so ``C`` has the same meaning across N and
    dimension.  ``repaired_scores`` are returned in the original mirrored
    endpoint order and are exactly induced by the fitted quadratic features.
    """
    if not (C > 0 and tolerance > 0 and (maximum_iterations > 0)):
        raise ValueError("Multiscale soft-decoder constants must be positive")
    system = complete_mirrored_svd(points, scores)
    feature = system["feature"]
    count = feature.shape[1]
    if schedule == "adjacent":
        lags = [1]
    elif schedule == "dyadic":
        lags = []
        lag = 1
        while lag < count:
            lags.append(lag)
            lag *= 2
    else:
        raise ValueError("schedule must be 'adjacent' or 'dyadic'")
    constraints = torch.cat([feature[:, lag:] - feature[:, :-lag] for lag in lags], dim=1)
    row_norm = constraints.norm(dim=-1, keepdim=True)
    if not bool(torch.isfinite(row_norm).all() and (row_norm > 0).all()):
        raise ArithmeticError("A multiscale comparison feature is degenerate")
    constraints /= row_norm
    batch, comparison_count, dimension = constraints.shape
    ranks = torch.linspace(-1, 1, count, device=feature.device, dtype=feature.dtype).expand(
        batch, -1
    )
    coefficient = (feature.transpose(-1, -2) @ ranks[..., None]).squeeze(-1)
    coefficient /= system["kernel_eigenvalues"]
    identity = torch.eye(dimension, device=coefficient.device, dtype=coefficient.dtype).expand(
        batch, -1, -1
    )
    weight = C / comparison_count
    iterations = torch.zeros(batch, device=coefficient.device, dtype=torch.int64)
    converged = torch.zeros(batch, device=coefficient.device, dtype=torch.bool)

    def derivatives(value):
        margin = torch.einsum("bnd,bd->bn", constraints, value)
        residual = (1 - margin).clamp_min(0)
        objective = 0.5 * value.square().sum(-1) + 0.5 * weight * residual.square().sum(-1)
        gradient = value - weight * torch.einsum("bnd,bn->bd", constraints, residual)
        active = (residual > 0).to(value.dtype)
        hessian = identity + weight * torch.einsum(
            "bni,bn,bnj->bij", constraints, active, constraints
        )
        return (objective, gradient, hessian)

    for iteration in range(maximum_iterations):
        objective, gradient, hessian = derivatives(coefficient)
        residual_norm = gradient.norm(dim=-1)
        threshold = tolerance * (1 + coefficient.norm(dim=-1))
        converged |= residual_norm <= threshold
        if bool(converged.all()):
            break
        step = torch.linalg.solve(hessian, gradient[..., None]).squeeze(-1)
        descent = (gradient * step).sum(-1).clamp_min(0)
        alpha = torch.ones(batch, device=coefficient.device, dtype=coefficient.dtype)
        accepted = converged.clone()
        candidate = coefficient.clone()
        for _ in range(60):
            trial = coefficient - alpha[:, None] * step
            trial_objective = derivatives(trial)[0]
            sufficient = trial_objective <= objective - 0.0001 * alpha * descent
            update = sufficient & ~accepted
            candidate[update] = trial[update]
            accepted |= sufficient
            if bool(accepted.all()):
                break
            alpha[~accepted] *= 0.5
        if not bool(accepted.all()):
            raise ArithmeticError("Multiscale soft decoder line search failed")
        coefficient = candidate
        iterations[~converged] = iteration + 1
    objective, gradient, _ = derivatives(coefficient)
    gradient_residual = gradient.norm(dim=-1)
    raw = torch.einsum("bp,bpd->bd", coefficient, system["center_derivative"])
    norm = raw.norm(dim=-1, keepdim=True)
    predicted_sorted = torch.einsum("bnp,bp->bn", feature, coefficient)
    repaired_scores = torch.zeros_like(predicted_sorted)
    repaired_scores.scatter_(1, system["order"], predicted_sorted)
    repaired_values = repaired_scores.sort(dim=-1).values
    repaired_gap = repaired_values.diff(dim=-1)
    gap_allowance = (
        128
        * torch.finfo(feature.dtype).eps
        * (1 + repaired_values.abs().amax(dim=-1, keepdim=True))
    )
    strict_repair = (repaired_gap > gap_allowance).all(-1)
    certified = (
        torch.isfinite(coefficient).all(-1)
        & torch.isfinite(raw).all(-1)
        & (norm[:, 0] > 0)
        & strict_repair
        & (gradient_residual <= 100 * tolerance * (1 + coefficient.norm(dim=-1)))
    )
    if not bool(torch.isfinite(norm).all() and (norm > 0).all()):
        raise ArithmeticError("Multiscale soft decoder returned a zero gradient")
    return dict(
        direction=raw / norm,
        coefficients=coefficient,
        repaired_scores=repaired_scores,
        repaired_gap=repaired_gap.amin(-1),
        certified=certified,
        iterations=iterations,
        objective=objective,
        gradient_residual=gradient_residual,
        penalty_C=coefficient.new_full((batch,), C),
        comparison_count=coefficient.new_full((batch,), comparison_count, dtype=torch.int64),
        lags=lags,
        feature_dimension=coefficient.new_full((batch,), dimension, dtype=torch.int64),
    )


def robust_quadratic_centers(points, scores, fallback_C=100000.0, fallback_schedule="dyadic"):
    """Use hard AC/VC when certified and repair only hard-cone failures.

    The fallback constants were selected on the frozen development cohort and
    confirmed on independent targets.  A successful batched hard solve is
    returned without any copying or recomputation, preserving exact-quadratic
    behavior.  When the batch is not wholly feasible, targets are checked
    separately and only failed targets receive multiscale ranking repair.
    """
    try:
        hard, information = quadratic_centers(points, scores)
        if not bool(hard["certified"].all()):
            raise ArithmeticError("Uncertified batched hard center")
        return (
            hard,
            dict(
                hard_success=torch.ones(len(points), device=points.device, dtype=torch.bool),
                fallback_used=torch.zeros(len(points), device=points.device, dtype=torch.bool),
                per_target=[{"mode": "hard", "information": information}] * len(points),
                fallback_C=fallback_C,
                fallback_schedule=fallback_schedule,
            ),
        )
    except (ArithmeticError, RuntimeError, AssertionError):
        pass
    directions = points.new_zeros((len(points), 2, points.shape[-1]))
    hard_success = torch.zeros(len(points), device=points.device, dtype=torch.bool)
    certified = torch.zeros(len(points), 2, device=points.device, dtype=torch.bool)
    records = []
    for target in range(len(points)):
        point = points[target : target + 1]
        score = scores[target : target + 1]
        try:
            result, information = quadratic_centers(point, score)
            if not bool(result["certified"].all()):
                raise ArithmeticError("Uncertified hard center")
            mode = "hard"
            hard_success[target] = True
            detail = information
        except (ArithmeticError, RuntimeError, AssertionError) as hard_error:
            soft = quadratic_multiscale_soft_direction(
                point, score, C=fallback_C, schedule=fallback_schedule
            )
            if not bool(soft["certified"].all()):
                raise ArithmeticError("Uncertified robust ranking repair") from hard_error
            result, information = quadratic_centers(point, soft["repaired_scores"])
            if not bool(result["certified"].all()):
                raise ArithmeticError("Uncertified repaired hard center") from hard_error
            mode = "repaired"
            detail = {
                "hard_error": str(hard_error),
                "soft_iterations": soft["iterations"].cpu().tolist(),
                "soft_gradient_residual": soft["gradient_residual"].cpu().tolist(),
                "repaired_gap": soft["repaired_gap"].cpu().tolist(),
                "information": information,
            }
        directions[target] = result["direction"][0]
        certified[target] = result["certified"][0]
        records.append({"mode": mode, "information": detail})
    if not bool(certified.all()):
        raise ArithmeticError("Robust quadratic decoder returned an uncertified target")
    return (
        dict(direction=directions, certified=certified),
        dict(
            hard_success=hard_success,
            fallback_used=~hard_success,
            per_target=records,
            fallback_C=fallback_C,
            fallback_schedule=fallback_schedule,
        ),
    )


def quadratic_scores(points, gradient, hessian):
    return (
        torch.asinh(
            torch.einsum("bnd,bd->bn", points, gradient)
            + 0.5 * torch.einsum("bni,bij,bnj->bn", points, hessian, points)
        )
        + 2
    )


def angular_mse(direction, truth):
    direction = direction / direction.norm(dim=-1, keepdim=True)
    truth = truth / truth.norm(dim=-1, keepdim=True)
    dot = (direction * truth).sum(-1).clamp(-1, 1)
    return torch.atan2((direction - dot[..., None] * truth).norm(dim=-1), dot).square()


def effective_curvature(gradient, hessian, radius=1.0):
    d = gradient.shape[-1]
    trace = hessian.diagonal(dim1=-2, dim2=-1).sum(-1)
    aniso = hessian - trace[:, None, None] / d * torch.eye(
        d, device=hessian.device, dtype=hessian.dtype
    )
    return radius * aniso.norm(dim=(-1, -2)) / (math.sqrt(2 * (d + 2)) * gradient.norm(dim=-1))


def block_errors(direction, truth, basis):
    d = truth.shape[-1]
    local = (basis @ direction[..., None]).squeeze(-1)
    actual = (basis @ truth[..., None]).squeeze(-1)
    split = 2 * (d // 4)
    masses = torch.stack(
        (local[:, :split].square().sum(-1), local[:, split:].square().sum(-1)), -1
    )
    true_masses = torch.stack(
        (actual[:, :split].square().sum(-1), actual[:, split:].square().sum(-1)), -1
    )
    masses /= masses.sum(-1, keepdim=True)
    true_masses /= true_masses.sum(-1, keepdim=True)
    within = torch.stack(
        (
            angular_mse(local[:, :split], actual[:, :split]),
            angular_mse(local[:, split:], actual[:, split:]),
        ),
        -1,
    )
    return dict(
        energy_mse=(masses - true_masses).square().sum(-1),
        block_angular_mse=within,
        true_energy=true_masses,
        estimated_energy=masses,
    )
