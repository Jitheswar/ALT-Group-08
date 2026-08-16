# Resume Screening and Candidate Recommendation

A system that helps a Recruiter order a batch of Candidates against one Role, by first testing hard Requirements and then judging graded Fit.
It advises; it never decides.

The project's title uses "recommendation" informally for the second stage.
In the codebase that stage is **Ranking** and its output is a **Shortlist** - "recommendation" is not used as a term.

## Language

### The role side

**Role**:
A single open position that one Screening Run is conducted against.
_Avoid_: Job, Posting, Vacancy, Opening

**Job Description**:
The source document describing a Role, from which its Requirements are drawn.
_Avoid_: JD, Spec, Posting

**Requirement**:
A single hard condition of a Role that a Candidate either meets or does not, with no middle ground.
_Avoid_: Criterion, Qualification, Skill, Must-have

**Requirement Set**:
The Recruiter-approved list of Requirements that a Screening Run gates on.
A Requirement proposed by extraction is not part of the Set until the Recruiter approves it.
_Avoid_: Criteria, Checklist, Rubric

### The candidate side

**Candidate**:
A person under consideration for a Role, represented in the system by exactly one Resume.
_Avoid_: Applicant, Profile, Lead

**Resume**:
The document a Candidate supplies as their own account of their experience.
_Avoid_: CV, Profile, Document

**Redacted Resume**:
A Resume with direct identity signals removed, and the only form of a Resume that Ranking is ever shown.
_Avoid_: Anonymised Resume, Masked Resume, Blind Resume

**Qualified Candidate**:
A Candidate who met every Requirement of a Role and therefore passed into Ranking.
_Avoid_: Match, Eligible, Shortlisted

**Unresolved Candidate**:
A Candidate whose Screening produced no valid verdict, and who is therefore neither Qualified nor disqualified.
_Avoid_: Failed, Errored, Skipped

### The two stages

**Screening**:
The stage that tests a Candidate against every Requirement of a Role and returns a binary outcome with a citable reason.
_Avoid_: Filtering, Matching, Parsing, Sifting

**Fit**:
The graded, non-binary judgement of how well a Qualified Candidate suits a Role.
Never a pass or a fail.
_Avoid_: Score, Match, Relevance

**Ranking**:
The stage that orders Qualified Candidates by Fit.
_Avoid_: Scoring, Sorting, Recommending

### Outputs and actors

**Shortlist**:
The ordered list of Qualified Candidates that a Screening Run returns, each carrying a Justification.
_Avoid_: Results, Top-N, Recommendations

**Justification**:
The written reason attached to a Screening outcome or a Fit judgement, stating what in the Resume supports it.
_Avoid_: Explanation, Rationale, Reason

**Recruiter**:
The human who supplies the Role and the Resumes, reads the Shortlist, and makes every decision about a Candidate.
_Avoid_: User, Hiring Manager, Reviewer

**Screening Run**:
One execution of Screening then Ranking, over one Role and one batch of Resumes.
The unit of work and the unit of record.
_Avoid_: Job, Batch, Session, Search

### Evaluation

**Proxy Relevance**:
A relevance judgement inferred from a corpus's existing category labels rather than stated by a human, treating resumes sharing a Role's category as relevant and all others as not.
_Avoid_: Ground truth, Labels, Weak supervision

**Gold Set**:
The small set of Role-and-Resume pairs relevance-judged by hand against a written rubric, held out to test whether Proxy Relevance is misleading.
_Avoid_: Test set, Validation set, Benchmark

**Evaluation Role**:
A Role used by the evaluation harness rather than supplied by a Recruiter: a corpus category, the real Job Description matched to it, and its reviewed Requirement Set, held together under version control.
_Avoid_: Fixture, Test role, Synthetic role

**Counterfactual Sensitivity**:
The degree to which a Fit judgement moves when an identity signal in a Resume is altered and nothing else is.
_Avoid_: Bias, Fairness score, Disparate impact
