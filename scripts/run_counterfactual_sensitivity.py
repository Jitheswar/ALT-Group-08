"""Runs Counterfactual Sensitivity (ticket 09, ADR-0005) across every
checked-in Evaluation Role, on the same deepseek-v4-flash configuration
that produces reported results.

For each Evaluation Role, samples its Proxy-Relevant corpus Resumes,
alters one identity signal at a time in each (name, gender, nationality),
and scores both variants through screening.ranking.score_fit - the same
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
import os
import sys
from dataclasses import asdict
from pathlib import Path

from screening.counterfactual import run_counterfactual_sensitivity
from screening.deepseek_client import DeepSeekModelClient, build_live_transport
from screening.eval_roles import read_evaluation_roles
from screening.gold_set import exclude_gold_set_candidates, read_gold_set
from screening.proxy_relevance import CorpusResume

RESUME_ATLAS_DATASET = "ahmedheakl/resume-atlas"
API_KEY_ENV_VAR = "DEEPSEEK_API_KEY"


def load_resume_atlas_corpus(parquet_path: Path) -> list[CorpusResume]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["Category", "Text"])
    categories = table.column("Category").to_pylist()
    texts = table.column("Text").to_pylist()
    return [
        CorpusResume(candidate_id=f"resume-atlas-{index}", category=category or "", text=text)
        for index, (category, text) in enumerate(zip(categories, texts))
        if text and text.strip()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--resume-atlas-parquet", required=True, type=Path)
    parser.add_argument(
        "--evaluation-roles-dir", type=Path, default=Path("data/evaluation_roles/roles")
    )
    parser.add_argument("--gold-set", type=Path, default=Path("data/gold_set/gold_set.json"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Proxy-Relevant Resumes sampled per Evaluation Role; defaults to the module's own default",
    )
    args = parser.parse_args(argv)

    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        print(f"error: {API_KEY_ENV_VAR} must be set to run this measurement", file=sys.stderr)
        return 1

    evaluation_roles = read_evaluation_roles(args.evaluation_roles_dir)
    if not evaluation_roles:
        print(f"error: no Evaluation Roles found under {args.evaluation_roles_dir}", file=sys.stderr)
        return 1

    corpus = load_resume_atlas_corpus(args.resume_atlas_parquet)
    if not corpus:
        print(f"error: no Resumes loaded from {args.resume_atlas_parquet}", file=sys.stderr)
        return 1

    if args.gold_set.exists():
        gold_set = read_gold_set(args.gold_set)
        before = len(corpus)
        corpus = exclude_gold_set_candidates(corpus, gold_set)
        print(f"Held {before - len(corpus)} Gold Set Resume(s) out of the sample corpus")

    model_client = DeepSeekModelClient(build_live_transport(api_key))
    kwargs = {} if args.sample_size is None else {"sample_size": args.sample_size}
    report = run_counterfactual_sensitivity(evaluation_roles, corpus, model_client, **kwargs)

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
