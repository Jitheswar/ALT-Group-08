"""Compares the checked-in Gold Set's hand labels against Proxy Relevance
for the same pairs (ticket 08, ADR-0003) and writes the divergence report.

Needs no model client and no corpus download: the Gold Set already carries
its own Resume text, and the checked-in Evaluation Roles already carry their
own category, so this runs entirely from checked-in data.

With --rank-out, additionally runs the real Screening+Ranking pipeline over
each Evaluation Role's own Gold Set pool and reports rank metrics (NDCG@k,
Precision@k, MRR) two ways over the identical resulting order: against the
hand labels, and against Proxy Relevance for the same pool. That is the
rank-metric-to-rank-metric comparison the agreement rate above cannot show
on its own - it needs a live model client (--rank-out implies --live).

Usage:
    uv run python scripts/run_gold_set_eval.py \
        --out data/gold_set/gold-set-report.json
    uv run python scripts/run_gold_set_eval.py \
        --out data/gold_set/gold-set-report.json \
        --rank-out data/gold_set/gold-set-rank-report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from screening.deepseek_client import MissingApiKey, build_deepseek_client
from screening.eval_roles import read_evaluation_roles
from screening.gold_set import compare_gold_set_to_proxy_relevance, rank_gold_set, read_gold_set


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold-set", type=Path, default=Path("data/gold_set/gold_set.json"))
    parser.add_argument("--evaluation-roles-dir", type=Path, default=Path("data/evaluation_roles/roles"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--rank-out",
        type=Path,
        default=None,
        help=(
            "Path to also write the rank-metric comparison to. Requires a "
            "live model client (DEEPSEEK_API_KEY), since it runs the real "
            "Screening+Ranking pipeline over each Role's Gold Set pool"
        ),
    )
    args = parser.parse_args(argv)

    if not args.gold_set.exists():
        print(f"error: no Gold Set found at {args.gold_set}", file=sys.stderr)
        return 1

    gold_set = read_gold_set(args.gold_set)
    evaluation_roles = read_evaluation_roles(args.evaluation_roles_dir)
    if not evaluation_roles:
        print(f"error: no Evaluation Roles found under {args.evaluation_roles_dir}", file=sys.stderr)
        return 1

    report = compare_gold_set_to_proxy_relevance(gold_set, evaluation_roles)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(report), indent=2) + "\n")

    print(f"Compared {len(gold_set)} hand-labelled Gold Set pair(s) against Proxy Relevance:")
    print(
        f"  agreement-rate={report.agreement_rate:.1%} "
        f"proxy-false-positive-rate={report.proxy_false_positive_rate:.1%} "
        f"proxy-false-negative-rate={report.proxy_false_negative_rate:.1%}"
    )
    print(f"Wrote Gold Set comparison report to {args.out}")

    if args.rank_out is not None:
        try:
            model_client = build_deepseek_client(usage="to rank the Gold Set (--rank-out)")
        except MissingApiKey as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        rank_results = rank_gold_set(gold_set, evaluation_roles, model_client)
        count = len(rank_results)
        mean_hand_ndcg = sum(r.hand_ndcg_at_k for r in rank_results) / count
        mean_proxy_ndcg = sum(r.proxy_ndcg_at_k for r in rank_results) / count
        mean_hand_mrr = sum(r.hand_reciprocal_rank for r in rank_results) / count
        mean_proxy_mrr = sum(r.proxy_reciprocal_rank for r in rank_results) / count

        args.rank_out.parent.mkdir(parents=True, exist_ok=True)
        args.rank_out.write_text(
            json.dumps([asdict(r) for r in rank_results], indent=2) + "\n"
        )
        print(f"Ranked {count} Evaluation Role(s)' Gold Set pool(s):")
        print(
            f"  hand-NDCG={mean_hand_ndcg:.3f} proxy-NDCG={mean_proxy_ndcg:.3f}  "
            f"hand-MRR={mean_hand_mrr:.3f} proxy-MRR={mean_proxy_mrr:.3f}"
        )
        print(f"Wrote Gold Set rank comparison to {args.rank_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
