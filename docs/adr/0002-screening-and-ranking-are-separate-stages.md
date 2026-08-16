# Screening and Ranking are separate stages

Screening tests a Candidate against a Role's Requirements and returns a binary outcome with a citable reason; Ranking then orders the Qualified Candidates by graded Fit.
The obvious alternative - one LLM call per Resume emitting a single 0-100 score - was rejected because it makes a hard disqualification indistinguishable from a weak match, and leaves no answerable form of the question "why is this Candidate 40th?".
Splitting the stages costs more to build and is what makes the system evaluable rather than merely demonstrable: the two stages fail differently, so they are measured differently.

## Consequences

Requirements must exist as discrete, individually checkable items, not as prose.
Ranking never sees a Candidate who failed Screening, so Fit is only ever defined over Qualified Candidates.
A future engineer will be tempted to collapse these into one prompt for cost or latency reasons; that is a reversal of this decision, not an optimisation.
