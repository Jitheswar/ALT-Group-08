# We own output validation, because the provider does not enforce schemas

Screening verdicts and Fit rubrics are validated against Pydantic models on our side, retried twice, and then failed loudly - the affected Candidate becomes an Unresolved Candidate recorded in the Screening Run's JSONL, and parse-failure rate is reported alongside the rank metrics.

The model provider offers a JSON-object output mode but no strict schema enforcement, and its documentation states that the API may occasionally return empty content.
A future reader will see a validation-and-retry layer that looks like defensive programming around a solved problem, because on several other providers schema-conformant output *is* a solved problem; this records that it was not available here.

Failing loudly rather than degrading is required by [[0001-advisory-only-no-automated-rejection]]: a Candidate whose verdict never parsed must not quietly become a rejection, because the system is not permitted to drop anyone.
A repair call - feeding malformed output back for correction - was rejected as a third inference path with no test coverage and its own failure modes.

## Consequences

Unresolved Candidate is a first-class state in the domain, not an error condition, and it appears in the Shortlist output rather than being filtered out of it.
Parse-failure rate is a published reliability number, which means a provider or prompt change that degrades it is visible rather than silent.
If schema enforcement becomes available, this layer is the thing to delete - not to keep as belt-and-braces.
