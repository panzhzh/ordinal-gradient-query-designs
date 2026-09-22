"""Strict ranking feasibility initialization using CUDA SVD and a Clarabel sparse QP."""

import torch


@torch.no_grad()
def rank_feasible_start(a):
    assert a.is_cuda and a.dtype == torch.float64 and (a.ndim == 2)
    import clarabel
    import numpy as np
    from scipy import sparse

    u, s, v = torch.linalg.svd(a, full_matrices=False)
    keep = s > 64 * torch.finfo(s.dtype).eps * max(a.shape) * s[0]
    rank = int(keep.sum())
    assert rank > 0
    transform = v[:rank].T / s[:rank]
    b = a @ transform
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_threads = 1
    settings.max_iter = 250
    settings.tol_gap_abs = 1e-09
    settings.tol_gap_rel = 1e-09
    settings.tol_feas = 1e-09
    solver = clarabel.DefaultSolver(
        sparse.eye(rank, format="csc"),
        np.zeros(rank),
        sparse.csc_matrix(-b.cpu().numpy()),
        -np.ones(len(a)),
        [clarabel.NonnegativeConeT(len(a))],
        settings,
    )
    solution = solver.solve()
    if str(solution.status) not in ["Solved", "AlmostSolved"]:
        raise ArithmeticError(f"Convex phase I failed: {solution.status}")
    w = transform @ a.new_tensor(solution.x)
    w = w / w.norm()
    slack = a @ w
    allowance = 128 * torch.finfo(a.dtype).eps * (a.abs() @ w.abs())
    if not bool((slack > allowance).all()):
        raise ArithmeticError("Convex phase I lacks strict original-row margin")
    return dict(
        direction=w,
        history=[
            dict(
                solver="Clarabel",
                version=clarabel.__version__,
                status=str(solution.status),
                iterations=solution.iterations,
                minimum_original_margin=float(slack.min()),
                numerical_rank=rank,
                total_columns=a.shape[1],
                relative_smallest_singular=float(s[-1] / s[0]),
            )
        ],
    )
