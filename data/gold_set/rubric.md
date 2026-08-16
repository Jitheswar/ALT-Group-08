# Gold Set labelling rubric

This is the written rubric behind the Gold Set (ADR-0003, CONTEXT.md), checked
into version control so the labelling is reproducible and criticisable rather
than a private judgement call.

## What the Gold Set checks

Proxy Relevance judges a corpus Resume relevant to an Evaluation Role purely
by whether it shares the Role's corpus category.
That is cheap at corpus scale but it is a proxy, not a human judgement, and it
can be wrong in both directions: a Resume outside the category can still be a
genuinely relevant background, and a Resume inside the category can still be
an obviously wrong fit for the specific Role.

The Gold Set exists to catch exactly that.
For each pair below, a human reads the Role and the Resume and answers one
question, independent of the corpus category label on either side.

## The question being labelled

> Based only on the Resume's stated work history, skills, and domain, would a
> recruiter reasonably consider this Candidate on-topic for this Role - the
> kind of Resume worth reading further, as opposed to one from a plainly
> different field?

The answer is binary: **relevant** or **not relevant**.
There is no partial credit and no numeric scale, because the label has to be
directly comparable to Proxy Relevance's own binary relevant / not-relevant
judgement.

## What this is not

- **Not Screening.** The label does not ask whether the Candidate meets every
  hard Requirement in the Role's Requirement Set.
  A relevant Candidate can still be missing a specific Requirement (wrong
  years of experience, missing certification) and remain relevant here - the
  question is domain fit, not gate-passing.
- **Not Fit.** The label is not graded and does not rate how strong a match
  the Candidate is.
  A barely-relevant junior Resume and an outstanding senior Resume in the same
  field both label `relevant`.
- **Not a quality judgement on the Resume itself.** Formatting, length, and
  writing quality are ignored.

## Reading order

1. Read the Role's title and its full Requirement Set - the Requirements are
   the clearest signal of what the Role is actually hiring for, often sharper
   than the title alone (e.g. a title of "Engineer" backed by Requirements
   naming a specific discipline).
2. Read the Resume's stated job titles, responsibilities, and skills.
3. Judge domain overlap: does the Resume's professional background sit in the
   same field or a directly adjacent specialisation the Role is hiring into?
4. A close-but-different discipline (e.g. a general IT support background
   against a Network Security Engineer Role) is judged on whether the
   specific discipline overlaps in a way that would make the Resume worth a
   recruiter's attention - broad membership in the same industry alone
   (e.g. "both are in tech") is not sufficient on its own.
5. Record the label and a one-sentence Justification citing the specific
   Resume content the label rests on, in keeping with how a Justification is
   used everywhere else in this system (CONTEXT.md): what in the Resume
   supports the label.

## Provenance and sampling

Every pair is a real Resume from the checked-in Evaluation Roles' source
corpus (`ahmedheakl/resume-atlas`) against one of the 43 checked-in Evaluation
Roles.
Which pairs get labelled is chosen by `scripts/select_gold_set_candidates.py`,
deterministically rather than randomly, and covers three cases per Evaluation
Role (a fourth added for a subset to reach "roughly 150" from 43 Roles):

- **same_category** - a Resume sharing the Role's own corpus category, where
  Proxy Relevance says relevant.
- **adjacent_category** - a Resume from a different but related job family,
  the case most likely to expose a Proxy Relevance false negative.
- **distant_category** - a Resume from an unrelated job family, an
  easy-negative case included so the Gold Set is not composed entirely of
  hard cases.

The label is produced independently of which of these three buckets a pair
came from - the bucket is sampling metadata, not an input to the judgement in
the reading order above.

## Where the result lives

The finished, checked-in Gold Set is `data/gold_set/gold_set.json`
(`screening.gold_set.read_gold_set`).
Every entry there is one hand-labelled pair: the Evaluation Role it was
judged against, the Resume (its corpus id, its own corpus category, and its
text), the `relevant` label, and the Justification.
Gold Set candidate ids are held out of the corpus a full sweep runs against
(`screening.gold_set.exclude_gold_set_candidates`, wired into
`scripts/run_sweep.py`), so no Resume that was hand-labelled here is also
scored through Proxy Relevance at corpus scale in the same run.
`scripts/run_gold_set_eval.py` compares this file's hand labels against
Proxy Relevance for the same pairs and reports the divergence.
