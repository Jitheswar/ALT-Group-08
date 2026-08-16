# No vector prefilter in front of Screening

Every Candidate in a Screening Run is passed through Screening and, if Qualified, through Ranking.
There is deliberately no embedding index or BM25 stage narrowing the field first.

A reasonable reader will expect one - retrieval-then-rerank is the standard shape for this problem, and its absence looks like an oversight rather than a choice.
It is a choice: a prefilter is a second ranking system with its own failure modes bolted in front of the one being evaluated, and anything it discards never reaches the metrics, so its errors are invisible to every number this project reports.
At the volumes in scope (see [[0003-two-tier-evaluation-with-proxy-relevance]]) a model call per Candidate is affordable, so the prefilter would buy latency at the cost of measurability.

## Consequences

Cost and latency scale linearly with batch size, and that is accepted.
If a prefilter is added later it must be evaluated as part of the pipeline - its recall becomes a ceiling on every downstream metric, and reporting rank quality over only the survivors would overstate the system.
This decision is cheap to reverse; it is recorded because reversing it *silently* is the failure mode.
