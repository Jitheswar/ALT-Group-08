# A Requirement Set is only what the Recruiter approved

Requirements are extracted from the Job Description by an LLM, but extraction only ever *proposes*: nothing enters a Role's Requirement Set until the Recruiter reviews and approves it, and Screening gates strictly on the approved Set.
The alternative - letting extraction populate the gate directly - was rejected because it makes [[0002-screening-and-ranking-are-separate-stages]] decorative: a hallucinated Requirement would silently disqualify Candidates, and the "deterministic" gate would be exactly as reliable as an unsupervised extraction step.

## Consequences

There is a human approval step in the middle of an "automated" system, and it is load-bearing rather than ceremonial - the API and UI both have to model proposed-but-unapproved Requirements as a distinct state.
A Screening Run cannot begin until a Requirement Set is approved, which means the fully unattended batch run is not a supported mode.
Extraction quality can now be measured on its own, against what Recruiters actually approve and reject.
