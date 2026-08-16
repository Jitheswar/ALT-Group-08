# Two-tier evaluation: Proxy Relevance at scale, a hand-labelled Gold Set to check it

The system's output is an ordered Shortlist, so it is measured with rank metrics (NDCG@k, Precision@k, MRR) rather than classification accuracy over a fit label.
Relevance judgements at scale are constructed as **Proxy Relevance** from a category-labelled resume corpus - resumes sharing a Role's category count as relevant, all others do not - and validated against a **Gold Set** of roughly 150 Role-and-Resume pairs judged by hand against a written rubric.

The obvious cheaper path was rejected: several public resume-and-job-description datasets carry ready-made fit labels, but those labels are largely LLM-generated (`netsol/resume-score-details` states its 1,031 samples were produced and assessed with GPT-4o).
Scoring an LLM ranker against LLM-generated labels measures agreement with the labelling model, not correctness, and it produces a flattering number that means nothing.

## Consequences

Proxy Relevance is a weak signal and is named as such everywhere it appears; the Gold Set exists precisely because the proxy might be lying, so a divergence between the two is a finding rather than a bug.
An LLM judge may be reported as a third number, never as the headline metric.
Hand-labelling is the one part of this project that cannot be automated away, and the rubric used to produce the Gold Set is itself an artifact worth keeping under version control.
