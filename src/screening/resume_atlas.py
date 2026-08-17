"""Loads the resume-atlas corpus into CorpusResume rows, shared by every
script that sweeps or samples from it: scripts/run_sweep.py,
scripts/run_counterfactual_sensitivity.py, and
scripts/select_gold_set_candidates.py.

pyarrow is imported inside the function, not at module level, so importing
screening.resume_atlas does not pull pyarrow into the base package's import
graph - only scripts that actually load the corpus need the `data`
dependency group.
"""

from __future__ import annotations

from pathlib import Path

from screening.proxy_relevance import CorpusResume

RESUME_ATLAS_DATASET = "ahmedheakl/resume-atlas"


def load_resume_atlas_corpus(parquet_path: Path) -> list[CorpusResume]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["Category", "Text"])
    categories = table.column("Category").to_pylist()
    texts = table.column("Text").to_pylist()
    return [
        CorpusResume(candidate_id=f"resume-atlas-{index}", category=category or "", text=text)
        for index, (category, text) in enumerate(zip(categories, texts))
        if text and text.strip()
    ]
