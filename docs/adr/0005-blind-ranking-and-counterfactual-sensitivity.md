# Ranking sees only Redacted Resumes, and Counterfactual Sensitivity is a reported metric

Direct identity signals - name, gender markers, graduation and birth years, nationality, photo - are removed before a Resume reaches Ranking, so Fit is judged on a Redacted Resume.
Alongside that, **Counterfactual Sensitivity** is measured and reported as a first-class result: alter one identity signal, hold the rest of the Resume constant, and measure how far the Fit judgement moves.

Redaction is not free and is not a solution on its own.
It removes legitimate signal along with the identity signal, and it does not stop a model inferring demographics from context that survives redaction - schools, employers, phrasing.
It is adopted because it is cheap on day one and painful to retrofit, and because pairing it with a measurement is what turns a claim about fairness into evidence.

## Consequences

Redaction sits on the boundary between Screening and Ranking, so every path into Ranking has to go through it; there is no code path where Ranking sees an unredacted Resume.
The evaluation harness needs a counterfactual generator, and the reported result may well be unflattering - that is the point, and an unflattering number is a finding to publish rather than a bug to fix.
Screening may still see the unredacted Resume, since Requirements such as work authorization can depend on facts redaction would remove.
