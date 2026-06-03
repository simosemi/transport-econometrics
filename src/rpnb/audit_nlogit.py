"""Generate an RPNB/NLOGIT audit report."""

from __future__ import annotations

import argparse
from pathlib import Path


REPORT = """# RPNB/NLOGIT Audit Report

## Scope

This report documents the implementation choices in `rpnb` that matter when
benchmarking against NLOGIT random-parameters negative binomial models.

## Likelihood Formulation

- `rpnb` uses the NB2 negative binomial likelihood with
  `Var(y | mu) = mu + alpha * mu^2`.
- The log mean is `log(mu_i) = offset_i + X_i beta + Z_i b_g` when an offset is
  supplied.
- When no offset is supplied, `rpnb` uses a zero offset vector internally.
- The dispersion parameter is estimated as `alpha > 0` using an internal log
  transform.

## Alpha Parameterization

- `rpnb` reports `alpha` directly in the NB2 variance form.
- NLOGIT output may report an ancillary or transformed dispersion parameter
  depending on command syntax. Benchmarks should verify that NLOGIT values are
  converted to the same NB2 `alpha` scale before comparison.

## Random Parameter Generation

- Random parameters are normally distributed.
- Independent random parameters use `b_g = mean + draw_g * sd`.
- Correlated random parameters use a Cholesky factor, with transformed
  standard-normal draws mapped into coefficient draws.
- Continuous random variables and generated random categorical dummies each
  estimate both a mean and a standard deviation.

## Halton Draws

- `rpnb` supports pseudo-random, Halton, and Sobol draws.
- Halton draws are transformed to standard-normal draws before entering the
  random coefficient formula.
- Exact agreement with NLOGIT depends on sequence bases, skipping/leaping, draw
  ordering, scrambling, and panel grouping order. For audit comparisons, use the
  same number of draws and verify draw conventions where possible.

## Random Categorical Variables

- Fixed and random categorical variables are dummy-coded by `rpnb`.
- The declared reference category is dropped.
- Dummy names follow `variable_value`, for example `Interstate_0` when
  `Interstate` has reference `1`.
- Each non-reference random dummy receives `beta_random_mean[dummy]` and
  `beta_random_sd[dummy]`.
- NLOGIT syntax often requires users to create or declare equivalent dummy
  variables explicitly, so benchmarks should confirm identical reference
  categories and dummy ordering.

## Offset Handling

- If `offset` is supplied, `rpnb` adds it to `log(mu)` with coefficient fixed at
  1.
- If `offset` is omitted or set to `null`, no offset is used and the offset
  vector is zero.
- A supplied offset variable cannot also be listed as fixed or random.
- When no offset is supplied, a variable such as `log_exposure` may be modeled as
  an ordinary fixed or random covariate.

## Panel Likelihood

- When `group_id` is supplied, `rpnb` treats random coefficients as group-level
  draws and forms the simulated panel likelihood by multiplying probabilities
  within group before averaging over draws.
- When `group_id` is omitted, each observation acts as its own random-parameter
  group for random-parameter models.
- Comparisons with NLOGIT must use the same panel identifier and row ordering.

## Key Differences To Check Against NLOGIT

- NB2 dispersion scale and reported alpha transformation.
- Offset inclusion and whether the offset coefficient is fixed at 1.
- Dummy coding, reference categories, and category ordering.
- Random draw type, seed, sequence conventions, and number of draws.
- Whether random parameters are independent or correlated.
- Panel/group identifier and within-panel likelihood aggregation.
- Optimizer convergence quality and possible multiple local optima.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rpnb.audit_nlogit")
    parser.add_argument(
        "--out",
        default="nlogit_audit_report.md",
        help="Path for the generated markdown audit report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(REPORT, encoding="utf-8")
    print(f"Wrote NLOGIT audit report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
