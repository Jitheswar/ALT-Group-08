# Resume Screening and Candidate Recommendation

Orders a batch of candidates against one open role and explains every position in that order.
It advises; it never decides - there is no reject action, no threshold, and no cutoff anywhere in the system.

Screening happens in two separate stages:

1. **Screening** tests the hard requirements of a role.
   Every submitted candidate comes back either qualified or marked with the exact requirement they missed, each with a justification citing the resume.
2. **Ranking** grades the fit of the qualified candidates and produces a shortlist.
   Ranking runs on redacted resumes, so identity details are not visible to the model that orders people.

Requirements are proposed from the job description by a model, but nothing enters a role's requirement set until a recruiter reviews and approves it.

See [CONTEXT.md](./CONTEXT.md) for the vocabulary and [docs/adr/](./docs/adr) for the decisions behind the design.

## Running the demo

```bash
./start.sh              # scripted model client, no network and no API key
./start.sh --live       # real provider, needs DEEPSEEK_API_KEY
PORT=8080 ./start.sh    # override the port (default 5000)
```

The web UI walks through the whole flow: paste a job description, review and approve the extracted requirements, add resumes (text or PDF), then run the screening and read the shortlist.
Demo presets under `data/demo_presets/` prefill a job description and a batch of resumes for 40 of the evaluation roles, and everything stays editable afterwards.

## Command line

```bash
uv run screening extract \
  --job-description jd.txt \
  --out proposed.json

uv run screening screen \
  --role-id role-1 --title "Python Developer" \
  --proposed proposed.json \
  --approved approved.json \
  --resumes resumes/
```

`extract` proposes requirements for review; edit them into the approved file yourself.
`screen` refuses to run against a requirement set that was never approved.
Both default to the scripted client and take `--live` to use the real provider.

Run records are written append-only, one file per run, immutable once complete.

## Evaluation

The system is measured rather than demonstrated, against 43 evaluation roles built from real postings and a hand-labelled gold set:

```bash
uv run scripts/run_sweep.py                       # proxy relevance sweep and rank metrics
uv run scripts/run_gold_set_eval.py               # agreement with the hand-labelled gold set
uv run scripts/run_counterfactual_sensitivity.py  # ranking stability under identity swaps
```

Public datasets with ready-made fit labels were deliberately not used for the headline numbers - most of those labels are model-generated, so scoring against them measures agreement with the labelling model rather than correctness.

## Development

```bash
uv sync
uv run pytest
uv run pyright
```

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).
