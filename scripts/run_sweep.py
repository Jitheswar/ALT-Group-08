"""Runs the full corpus sweep (ticket 07): every checked-in Evaluation
Role, swept over every Resume in the resume-atlas corpus, on the same
deepseek-v4-flash configuration that produces reported results.

There is no scripted-client mode here. A number reported as the headline
result that was produced by a different model than the one described
would misrepresent what was measured - the sweep's own correctness is
covered by tests/test_sweep.py against fakes instead.

The checked-in Gold Set (ticket 08) is excluded from the corpus before
sweeping, via screening.gold_set.exclude_gold_set_candidates, so a Resume
a human already hand-labelled is never also scored through Proxy Relevance
at corpus scale in the same run.

Usage:
    uv run --group data python scripts/run_sweep.py \
        --resume-atlas-parquet PATH \
        --out data/evaluation_roles/sweep-report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from screening.deepseek_client import MissingApiKey, build_deepseek_client
from screening.eval_roles import read_evaluation_roles_metadata
from screening.evaluation_inputs import GOLD_SET_HELP, load_evaluation_inputs
from screening.ranking import DEFAULT_TOP_BAND_SIZE
from screening.resume_atlas import RESUME_ATLAS_DATASET
from screening.sweep import run_sweep

DEFAULT_METADATA_PATH = Path("data/evaluation_roles/metadata.json")


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
        help=GOLD_SET_HELP.format(noun="sweep corpus"),
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--top-band-size",
        type=int,
        default=None,
        help="k for Precision@k and NDCG@k; defaults to Ranking's own top band size",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help=(
            "Path to the checked-in Evaluation Roles metadata, reporting "
            "reviewed_sample_size alongside the sweep's rank metrics "
            "(spec story 34); defaults to data/evaluation_roles/metadata.json "
            "if present. A path given explicitly must exist"
        ),
    )
    args = parser.parse_args(argv)

    try:
        model_client = build_deepseek_client(usage="to run the sweep")
    except MissingApiKey as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    inputs = load_evaluation_inputs(
        evaluation_roles_dir=args.evaluation_roles_dir,
        resume_atlas_parquet=args.resume_atlas_parquet,
        gold_set=args.gold_set,
        held_out_noun="sweep corpus",
    )
    if inputs is None:
        return 1
    evaluation_roles, corpus = inputs.evaluation_roles, inputs.corpus

    if args.metadata is not None and not args.metadata.exists():
        print(f"error: Evaluation Roles metadata not found at {args.metadata}", file=sys.stderr)
        return 1
    metadata_path = args.metadata or (DEFAULT_METADATA_PATH if DEFAULT_METADATA_PATH.exists() else None)
    metadata = read_evaluation_roles_metadata(metadata_path) if metadata_path is not None else None

    top_band_size = args.top_band_size if args.top_band_size is not None else DEFAULT_TOP_BAND_SIZE
    report = run_sweep(evaluation_roles, corpus, model_client, k=top_band_size)

    output = asdict(report)
    if metadata is not None:
        output["reviewed_sample_size"] = metadata.reviewed_sample_size
        output["total_evaluation_roles"] = metadata.total_evaluation_roles

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")

    print(
        f"Swept {len(evaluation_roles)} Evaluation Role(s) over {len(corpus)} Resume(s) "
        f"from {RESUME_ATLAS_DATASET}:"
    )
    print(
        f"  Proxy Relevance NDCG@{report.k}={report.mean_proxy_ndcg_at_k:.3f} "
        f"Proxy Relevance Precision@{report.k}={report.mean_proxy_precision_at_k:.3f} "
        f"Proxy Relevance MRR={report.proxy_mrr:.3f} "
        f"parse-failure-rate={report.parse_failure_rate:.1%}"
    )
    if metadata is not None:
        print(
            f"  reviewed-sample-size={metadata.reviewed_sample_size}/"
            f"{metadata.total_evaluation_roles} Evaluation Roles"
        )
    else:
        print("  reviewed-sample-size=unknown (no Evaluation Roles metadata found)")
    print(f"Wrote sweep report to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
