# Reproducing the manuscript

## Environment

The prepared release is checked with Python 3.12.12, PyTorch 2.10.0, NumPy 2.5.1, SciPy 1.17.0, Clarabel 0.11.1, and pytest 9.0.2 on an NVIDIA A100. Dependencies in `pyproject.toml` permit compatible versions; these are the versions used for the release checks.

Use an existing CUDA-enabled PyTorch environment:

```bash
pip install -e '.[test]'
python scripts/check_reference.py
pytest -q
```

PyTorch GPU selection follows `CUDA_VISIBLE_DEVICES`. Device indices passed to scripts are logical indices within that visibility. Solver replay stays on CUDA; file checks, saved-result aggregation, and Clarabel initialization are CPU operations. No cluster scheduler, external service, or account is required.

## What is distributed

`data/manifest.json` records relative paths, SHA-256 hashes, file sizes, and cohort identities. Input archives contain the exact float64 gradients, Hessians, and query bases or endpoints needed for replay. They use ordinary NumPy arrays and load with `allow_pickle=False`.

| Cohort | Independent targets | Dimensions | Role |
| :-- | --: | :-- | :-- |
| `main` | 768 | 10, 20, 50 | Main geometry and decoder comparisons |
| Main replicas 0 and 1 | 192 reused targets | 10, 20, 50 | Curvature and random-mirror diagnostics |
| `independent` | 128 | 10, 20 | Separate acquisition validation |
| `robustness` | 192 | 10, 20, 50 | Cubic mismatch evaluation; separate calibration population for linear RankSVM |
| Ambiguity ensemble | 256 constructed pairs | 8 | Cross-component scale distinguishability |

Groups contain eight targets. The identity `cohort/repR-streamS-regimeH-dD/T` identifies a target across all its radii, methods, and interventions. `regime=0` is positive definite; `regime=1` alternates eigenvalue signs. Streams are independent generation streams. Reusing a target at another radius does not increase the independent sample count.

Only the first three of the original query bases are distributed, because those are all the public experiments consume. There is no lossy quantization or change to those bases. Random-mirror directions and cubic ridge coefficients are frozen explicitly.

`data/reference/*.csv` contains target-level squared angular errors in **radians squared**. `results/paper/` contains the current manuscript's degree-valued tables and figure data. Values are kept at full precision. Unsuccessful hard fits are recorded through `hard_success`; they are not encoded as zero errors. Hybrid and direct-soft losses remain available for these targets.

## Experiment mapping

Run from the repository root. Every `--output` must be a new filename.

| Command | Evidence |
| :-- | :-- |
| `python scripts/reproduce.py main --output runs/main.jsonl` | Main 768-target AC/VC geometry comparisons and Table 1 decoder controls |
| `python scripts/reproduce.py curvature --output runs/curvature.jsonl` | Figure 2a, fixed radius and varying Hessian strength |
| `python scripts/reproduce.py random-mirrors --output runs/random.jsonl` | Figure 2b, matched budget and VC decoder |
| `python scripts/reproduce.py independent --output runs/independent.jsonl` | Independent acquisition comparisons in Figure 2c |
| `python scripts/reproduce.py ambiguity --output runs/ambiguity.jsonl` | Table 2 VC recovery and ranking separation |
| `python scripts/reproduce.py mismatch --output runs/mismatch.jsonl` | Frozen hybrid, direct-soft, and rank decoders under cubic mismatch |
| `python examples/scale_resolution.py` | Figure 1b's four strict-ranking intervals versus one |
| `python scripts/reproduce.py linear-calibration --output runs/calibration.jsonl` | Selection of the linear penalty from the separate 192-target population |

The main run includes unshifted cycle/two-cycle controls. The paper's independent phase contrast is an auxiliary analysis; the default independent runner reproduces its three acquisition geometries. General phase construction is available through `make_queries(..., phase=...)`.

A first replay can use `--dimension 10 --max-groups 1`. The result file marks this as a subset. Such output should not be compared as a pooled 192- or 768-target result.

```bash
python scripts/reproduce.py curvature --dimension 10 --max-groups 1 \
  --device cuda:0 --output runs/smoke.jsonl
python scripts/summarize.py runs/smoke.jsonl
```

`scripts/summarize.py` matches reference rows by target identity and condition, prints maximum saved-loss differences, and reports per-condition RMSE, median, and p90. Solver/BLAS versions can lead to small floating-point differences; convergence certificates and aggregate accuracy should agree. Reference files are never overwritten.

## Parallel replay

Shard indices select disjoint frozen groups. On two allocated, visible GPUs:

```bash
python scripts/reproduce.py main --device cuda:0 --shards 2 --shard-index 0 \
  --threads 1 --output runs/main-0.jsonl &
python scripts/reproduce.py main --device cuda:1 --shards 2 --shard-index 1 \
  --threads 1 --output runs/main-1.jsonl &
wait
python scripts/summarize.py runs/main-0.jsonl runs/main-1.jsonl
```

Assign CPU resources to the workers according to your scheduler allocation. A single visible GPU uses the default unsharded command. Workloads write one JSONL file per process. Duplicate target-condition records across input shards are rejected by the summary script.

## Statistics

For each target, average squared angular errors over its evaluated radii; then average over targets. Convert the square root to degrees for angular RMSE. Relative MSE reduction is

\[
1-\frac{L_{\mathrm{cycle}}}{L_{\mathrm{reference}}}.
\]

Pair targets by their full identifiers. Bootstrap whole targets within **stream × Hessian-regime × dimension** strata, retaining all radius measurements together. The release uses 32,768 draws and the source analysis's random seeds. The post-hoc Frames/two-cycle comparison family uses 97.5% intervals; random mirrors uses 95%.

```bash
python scripts/paired_bootstrap.py --population original-matched \
  --reference frames --device cuda:0
python scripts/paired_bootstrap.py --population independent-validation \
  --reference staggered-two --device cuda:0
python scripts/paired_bootstrap.py --population random-mirrors \
  --reference random-mirrors --device cuda:0
```

The matched original population is 512 targets at `d=10,20`, using only `r=1,10`. The independent population has 128 targets under those same conditions. A confidence interval from one population must not be attached to the other's point estimate. Archived figure CSVs retain the exact reported intervals. In the independent cohort, cycle versus Frames gives a 19.00% MSE reduction [1.44%, 32.22%]; cycle versus equally staggered two-cycle gives −9.61% [−31.58%, 7.46%]. Both intervals are 97.5%. The latter contrast does not establish a stable mean-accuracy gain from connectivity. The ambiguity experiment instead measures distinguishability for its defined target family.

## Curvature and penalty calibration

At radius one, use the original gradient/query basis and replace `H` with `γH`, for `γ ∈ {0, 0.1, 1, 10}`. The linear decoder's penalty is selected by minimum pooled MSE over those four levels on a **separate** 192-target cohort, from `10⁻²,…,10⁶`. The frozen value is `10⁴`.

The fallback uses dyadic comparisons and `C=10⁵`, selected on its development targets and evaluated on the separate robustness cohort. Replaying the mismatch experiment applies these frozen settings; it does not select them again using evaluation losses.

## Cubic remainder and repair analysis

The cubic term is a weighted sum of four cubic ridge functions. It has zero gradient at the center. At the queried endpoints its centered RMS is rescaled to 0%, 1%, 3%, or 10% of the quadratic score's centered RMS. The same ridge coefficients are used across all strengths for a target.

At 3%, 127/192 targets pass hard recovery. Among the remaining 65, 63 have dimension 50. The frozen results are:

| Decoder on the 65 hard failures | Angular RMSE | Median | p90 |
| :-- | --: | --: | --: |
| Repair → VC | 1.692° | 1.673° | 1.814° |
| Direct soft quadratic | 1.679° | 1.657° | 1.795° |
| Rank weighting | 8.501° | 8.108° | 11.227° |

The repair changes a median 0.326% of unordered pair relations on these failures. Both soft outputs are included: the re-decoding step establishes consistency with the repaired order, while the direct fit also supplies an accurate direction. See `results/paper/repair_subsets.json` for all three nonzero mismatch levels.

## Ambiguity ensemble

The frozen 256 pairs have block-separated quadratic offsets and different positive component scales. Two-cycle rankings are identical within each pair. The cycle separates all 256. Each pair's half-angle is a lower bound on any deterministic two-cycle decoder's equal-pair angular RMSE. Compare each connected pair RMSE to **its own bound**, before aggregating; 213/256 connected VC pairs fall below that bound.

Figure 1 uses a separate, fixed one-parameter family. Its plot sampling points are not additional independent targets. The three analytic crossing values divide the interval into four strict-ranking states, with ties excluded at the crossings.
