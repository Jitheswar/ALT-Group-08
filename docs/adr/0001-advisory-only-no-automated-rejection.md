# Advisory only: the system never rejects a Candidate

A Screening Run produces an ordered Shortlist with a Justification for every outcome, and a Recruiter makes every decision about every Candidate.
The system applies no cutoff and discards nobody: Candidates who fail Screening are still returned, marked with the Requirement they missed, rather than dropped.
We chose this over an auto-reject cutoff because an advisory ranker can be wrong without harming anyone, can be evaluated against human judgement, and stays outside the regulatory category that automated employment decision tools fall into in several jurisdictions.

## Consequences

The API has no "reject" verb and no configurable threshold, and the UI has no view that hides Candidates from the Recruiter.
Every Screening outcome and Fit judgement must carry a Justification, because a ranking a Recruiter cannot interrogate is not advice.
If automated rejection is ever wanted, it is a new decision that supersedes this one, not a config change.
