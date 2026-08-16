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
import os
import sys
from dataclasses import asdict
from pathlib import Path

from screening.deepseek_client import DeepSeekModelClient, build_live_transport
from screening.eval_roles import read_evaluation_roles
from screening.gold_set import exclude_gold_set_candidates, read_gold_set
from screening.proxy_relevance import CorpusResume
from screening.sweep import run_sweep

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
        "--top-band-size",
        type=int,
        default=None,
        help="k for Precision@k and NDCG@k; defaults to Ranking's own top band size",
    )
    args = parser.parse_args(argv)

    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        print(f"error: {API_KEY_ENV_VAR} must be set to run the sweep", file=sys.stderr)
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
        print(f"Held {before - len(corpus)} Gold Set Resume(s) out of the sweep corpus")

    model_client = DeepSeekModelClient(build_live_transport(api_key))
    kwargs = {} if args.top_band_size is None else {"k": args.top_band_size}
    report = run_sweep(evaluation_roles, corpus, model_client, **kwargs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(report), indent=2) + "\n")

    print(
        f"Swept {len(evaluation_roles)} Evaluation Role(s) over {len(corpus)} Resume(s) "
        f"from {RESUME_ATLAS_DATASET}:"
    )
    print(
        f"  NDCG@{report.k}={report.mean_ndcg_at_k:.3f} "
        f"Precision@{report.k}={report.mean_precision_at_k:.3f} "
        f"MRR={report.mrr:.3f} "
        f"parse-failure-rate={report.parse_failure_rate:.1%}"
    )
    print(f"Wrote sweep report to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
