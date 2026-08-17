"""Runs Counterfactual Sensitivity (ticket 09, ADR-0005) across every
checked-in Evaluation Role, on the same deepseek-v4-flash configuration
that produces reported results.

For each Evaluation Role, samples its Proxy-Relevant corpus Resumes,
alters one identity signal at a time in each (name, gender, nationality),
and judges both variants through screening.ranking.judge_fit - the same
redact-then-rank path real Ranking calls go through. The report carries
both the Fit-movement measurement and the deterministic Redaction-leak
check side by side; neither is filtered or thresholded, so an unflattering
number here is reported as-is rather than smoothed over (ADR-0005).

Usage:
    uv run --group data python scripts/run_counterfactual_sensitivity.py \
        --resume-atlas-parquet PATH \
        --out data/evaluation_roles/counterfactual-sensitivity-report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from screening.counterfactual import DEFAULT_SAMPLE_SIZE, run_counterfactual_sensitivity
from screening.deepseek_client import MissingApiKey, build_deepseek_client
from screening.evaluation_inputs import GOLD_SET_HELP, load_evaluation_inputs
from screening.resume_atlas import RESUME_ATLAS_DATASET


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--resume-atlas-parquet", required=True, type=Path)
    parser.add_argument(
        "--evaluation-roles-dir", type=Path, default=Path("data/evaluation_roles/roles")
    )
    parser.add_argument(
        "--gold-set",
        type=Path,
        default=None,
        help=GOLD_SET_HELP.format(noun="sample corpus"),
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Proxy-Relevant Resumes sampled per Evaluation Role; defaults to the module's own default",
    )
    args = parser.parse_args(argv)

    try:
        model_client = build_deepseek_client(usage="to run this measurement")
    except MissingApiKey as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    inputs = load_evaluation_inputs(
        evaluation_roles_dir=args.evaluation_roles_dir,
        resume_atlas_parquet=args.resume_atlas_parquet,
        gold_set=args.gold_set,
        held_out_noun="sample corpus",
    )
    if inputs is None:
        return 1
    evaluation_roles, corpus = inputs.evaluation_roles, inputs.corpus

    sample_size = args.sample_size if args.sample_size is not None else DEFAULT_SAMPLE_SIZE
    report = run_counterfactual_sensitivity(
        evaluation_roles, corpus, model_client, sample_size=sample_size
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(report), indent=2) + "\n")

    total_pairs = sum(len(result.measurements) for result in report.results)
    print(
        f"Measured Counterfactual Sensitivity over {total_pairs} counterfactual pair(s) "
        f"across {len(evaluation_roles)} Evaluation Role(s) from {RESUME_ATLAS_DATASET}:"
    )
    print(
        f"  mean-fit-weight-movement={report.mean_fit_weight_movement:.2f} "
        f"max-fit-weight-movement={report.max_fit_weight_movement} "
        f"redaction-leak-rate={report.redaction_leak_rate:.1%}"
    )
    print(f"Wrote Counterfactual Sensitivity report to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
