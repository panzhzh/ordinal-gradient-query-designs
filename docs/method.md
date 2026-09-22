# Method and API reference

## Observation model

At a fixed center, a local quadratic has gradient `g` and symmetric Hessian `H`:

\[
f(y)=g^\top y+\tfrac12 y^\top H y.
\]

The oracle returns a strict complete ranking of a batch of offsets. Any strictly increasing transformation of the scores produces the same input. `recover_direction(points, order)` accepts only coordinates and an ascending permutation. Ground-truth coefficients appear in synthetic oracle generation and evaluation, never in decoding or initialization.

All recovered directions point **toward increasing objective values**. Negate the direction for minimization.

## Query construction

The staggered cycle allocates four lines to each primary matching edge and two lines to each bridge edge. Across `d` coordinates, this produces `3d` lines and `6d` antipodal endpoints. The primary phase is `π/8`. All structured controls share the same orthogonal coordinate basis.

The returned layout is `[z_1, …, z_3d, −z_1, …, −z_3d]`. Keep that layout when passing coordinates to the decoder; `order` specifies the separate ranking permutation. Offsets, rather than absolute locations, are the coordinates relative to the gradient-estimation center. For example, send `center + points` to your oracle, then decode using `points`.

`make_queries` supports an even dimension at least eight, batches, physical radius, a reproducible local seed, and explicitly supplied orthogonal bases. `frames` uses three independent bases. `random-mirrors` uses independent normalized Gaussian directions. `graphs.py` additionally provides support-graph and grid utilities for examining other allocations; the paper configuration fixes the evaluated designs.

## Exact physical feature metric

The decoder factorizes the centered polynomial kernel

\[
K_c=C\big[(1+YY^\top)^{\circ 2}\big]C,
\qquad C=I-\mathbf1\mathbf1^\top/N.
\]

Odd and even feature blocks are separated before singular value decomposition. Retained singular values remain in the feature map. The coefficient-to-gradient derivative is mapped back through the same factorization; coordinates of an arbitrary SVD basis are never interpreted directly as gradient components.

The SVD cutoff is `64 × machine_epsilon × max(matrix_shape) × largest_singular_value`. The implementation verifies kernel reconstruction. The cycle's complete visible model has dimension `3d−1`; Frames and random-mirror controls retain their entire visible spans as well. Physical coordinates preserve the paper's relative linear/quadratic coefficient metric. Unit-radius normalization would define a different prior and is not silently applied.

## Hard centers

For features in ascending order, normalized adjacent differences are rows of `A`. The strict cone is `Aw > 0`. The analytic center minimizes

\[
\tfrac12\|w\|^2-\sum_i\log(a_i^\top w).
\]

VC minimizes

\[
\tfrac12\|w\|^2+\tfrac12\log\det(I+B^\top B),
\qquad B_i=a_i/(a_i^\top w).
\]

Initialization first uses the ordinal rank ramp. If it is not strictly feasible, `feasibility.py` solves a reduced convex phase-I problem with Clarabel. No score gaps enter that problem. AC initializes VC. The current center routine computes both centers; selecting `quadratic-ac` selects its AC output and does not constitute a separate optimized timing path.

The batched CUDA solvers retain the paper's convergence checks and float64 arithmetic. Clarabel's sparse phase-I solve is a CPU support operation with one thread. Failures raise an exception at the public API rather than producing a silently accepted direction. A batch must share a visible feature rank; use separate calls for different geometries or dimensions.

## Linear and soft quadratic controls

The full-ranking linear control uses normalized adjacent coordinate differences:

\[
\tfrac12\|v\|^2+\tfrac C2\sum_i[1-\tilde a_i^\top v]_+^2,
\qquad C=10^4.
\]

The soft quadratic model compares rank positions at lags `1, 2, 4, …`, using normalized **quadratic** differences:

\[
\tfrac12\|w\|^2+\frac{C}{2|\mathcal P|}
\sum_{(i,j)\in\mathcal P}[1-a_{ij}^\top w]_+^2,
\qquad C=10^5.
\]

One objective sums its losses; the other averages them. Their penalties are therefore not directly interchangeable.

`soft-quadratic` returns the gradient from this fit. `hybrid-vc` first attempts hard decoding; only failed targets receive the soft fit, a new order induced by its fitted utilities, and a hard decode of that order. Successful hard targets bypass repair. The repair and solver diagnostics remain available in `Recovery.diagnostics`.

A hard-center certificate verifies the numerical center and strict consistency with its input order. After repair, that order is the **repaired order**. It is not a certificate of the true gradient. The direct-soft comparison in the paper is included in the reference data and mismatch runner.

## Main entry points

```python
make_queries(dimension, *, batch_size=1, radius=1.0,
             geometry="staggered-cycle", seed=0, device="cuda",
             bases=None, phase=math.pi/8)

recover_direction(points, order, *, method="quadratic-vc",
                  linear_C=1e4, repair_C=1e5)
```

`points`: CUDA float64 `[batch, endpoints, dimension]`.

`order`: CUDA int64 `[batch, endpoints]`; each row is a permutation, smallest value first.

`Recovery.direction`: unit ascent directions `[batch, dimension]`.

`Recovery.certified`: per-target Boolean convergence checks.

`Recovery.diagnostics`: feature dimensions, condition numbers, iteration and timing information, or hard/repair decisions, depending on the method.
