# Evaluation Roles use real postings, never Job Descriptions derived from the corpus

Each Evaluation Role pairs a `resume-atlas` category with a real Job Description drawn from a public postings corpus and matched by script, plus a Requirement Set produced by extraction and reviewed on a sample of at least 10 of the 43 before being checked in.
Job Descriptions are never generated from resumes in the corpus.

That prohibition is the entire point of this record, because violating it is invisible afterwards.
Generating a Role's Job Description from sampled resumes in its category produces a document that describes those resumes, so the system is then ranking Candidates against a text derived from the Candidates.
Rank metrics come out excellent and mean nothing - the same circularity [[0003-two-tier-evaluation-with-proxy-relevance]] rejects in LLM-generated relevance labels, moved one step upstream where it is much harder to spot.
Synthesizing a Job Description from the bare category *label* carries no such path and is an acceptable fallback; deriving one from corpus content is not.

The sample review satisfies [[0004-requirement-sets-are-recruiter-approved]] for the evaluation path, where no Recruiter is present.
Auto-approving whatever extraction produced would have made that decision decorative in exactly the mode that generates published results.

## Consequences

Regenerating Evaluation Roles invalidates comparison against every previously recorded run, so the checked-in set is the unit of reproducibility and changing it is a versioned event.
The reviewed sample size is reported alongside the metrics; a reader can discount the results accordingly rather than having to assume.
The category-to-posting mapping is a reviewable artifact in its own right - a bad match silently corrupts Proxy Relevance for that Role, and nothing downstream would reveal it.
