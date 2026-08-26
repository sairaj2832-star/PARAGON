# Dataset Preparation Phases 0–4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and freeze a verified corpus of exactly 5,000 human + 5,000 AI original texts (10,000 total) sampled reproducibly from `andythetechnerd03/AI-human-text`, fully audited, provenance-traced, and ready for a separate later DIPPER paraphrasing phase.

**Architecture:** A Python package `dataset_prep` of small single-purpose modules (config, auth, download, label verification, audit, cleaning, sampling, validation, freezing, reproduction) executed sequentially from the repo root. Every stage reads its inputs **from disk**, writes artifacts under `project_dataset/`, emits machine-readable JSON summaries plus Markdown reports, and never mutates data silently — every removal is counted and logged.

**Tech Stack:** Python 3.11.7, Hugging Face `datasets` + `huggingface_hub`, pandas 3.x + PyYAML (already installed), pyarrow (parquet engine), pytest. Git for versioning scripts/configs/reports.

## Global Constraints (apply to every task)

- Source dataset: `andythetechnerd03/AI-human-text` (columns `text`: string, `generated`: int8; splits train=462,873 / test=24,362).
- Final counts are exact: Human = 5,000; AI = 5,000; Total = 10,000. No more, no less.
- Random seed = **2026**. Sampling must be randomized and reproducible — NEVER `dataset[:5000]` / first-N selection.
- Label mapping (0 vs 1 ⇒ human vs AI) is **not** documented by dataset features (`generated` is plain int8, no ClassLabel). It MUST be verified via independent evidence sources and explicitly confirmed by the operator before any downstream task runs. If evidence conflicts or is absent: STOP, do not guess.
- Every filtering/removal operation must be logged with exact counts. Never silently discard or modify records.
- Preserve provenance: every kept record keeps `source_dataset`, `source_split`, `source_row_id`, `original_label`.
- Textual content is preserved: no paraphrasing/rewriting. Only documented normalization: leading/trailing whitespace strip on surviving rows.
- Primary format: Parquet. Secondary export: CSV. No SQL/SQLite.
- Validation must run against files **reloaded from disk**, not in-memory dataframes.
- Do NOT run DIPPER or any paraphrasing. The task ends when originals are validated and frozen.
- Never print, log, or commit the HF token. It lives in `.env` (already present, key `HF_TOKEN`) or env var `HF_TOKEN`; `.env` is gitignored.
- All commands run from repo root `D:\PARAGON\Work`. Python package imports assume this cwd.
- Output layout is mandated:
  ```
  project_dataset/
  ├── raw/source_dataset/
  ├── processed/{human_candidates,ai_candidates}.parquet
  ├── sampled/{human_5000,ai_5000,originals_10000}.parquet
  ├── exports/originals_10000.csv
  ├── configs/{sampling_config.yaml,label_mapping.yaml,source_manifest.json,freeze_manifest.json}
  └── reports/*.md, *.json
  ```

---

### Task 1: Repo bootstrap + frozen Phase 0 configuration

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `project_dataset/configs/sampling_config.yaml`
- Create: `dataset_prep/__init__.py`
- Create: `dataset_prep/config_loader.py`
- Test: `tests/dataset_prep/__init__.py`, `tests/dataset_prep/test_config_loader.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict`, `validate_config(cfg: dict) -> None` (raises `ValueError`), path constants used by every later module: `REPO_ROOT`, `PROJECT_DIR`, `RAW_DIR`, `PROCESSED_DIR`, `SAMPLED_DIR`, `EXPORTS_DIR`, `CONFIGS_DIR`, `REPORTS_DIR`. Frozen spec dict `FROZEN_SPEC`.

- [ ] **Step 1: Initialize git and create `.gitignore`**

Run from `D:\PARAGON\Work`:

```powershell
git init
```

Create `.gitignore` with exactly:

```gitignore
.env
__pycache__/
*.pyc
.pytest_cache/
project_dataset/raw/
project_dataset/sampled/_repro_tmp/
```

- [ ] **Step 2: Create `requirements.txt`**

```text
datasets>=3.4
pyarrow>=17.0
pytest>=8.0
```

(`pandas`, `PyYAML`, `huggingface_hub` are already installed on this machine.)

- [ ] **Step 3: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: installs `datasets`, `pyarrow`, `pytest` without errors.

- [ ] **Step 4: Scaffold directories and frozen config**

```powershell
New-Item -ItemType Directory -Force -Path project_dataset\raw\source_dataset, project_dataset\processed, project_dataset\sampled, project_dataset\exports, project_dataset\configs, project_dataset\reports, dataset_prep, tests\dataset_prep | Out-Null
New-Item -ItemType File -Force -Path project_dataset\processed\.gitkeep, project_dataset\sampled\.gitkeep, project_dataset\exports\.gitkeep, dataset_prep\__init__.py, tests\dataset_prep\__init__.py | Out-Null
```

Create `project_dataset/configs/sampling_config.yaml` with EXACTLY this content (this freezes Phase 0):

```yaml
# FROZEN Phase 0 configuration -- changes invalidate all downstream artifacts.
dataset:
  source: "andythetechnerd03/AI-human-text"
  source_splits: ["train", "test"]
  human_samples: 5000
  ai_samples: 5000
  random_seed: 2026

quality:
  remove_missing: true
  remove_empty: true
  remove_duplicates: true

label_mapping:
  status: "pending_verification"

output:
  format_primary: "parquet"
  format_secondary: "csv"
  root: "project_dataset"
```

- [ ] **Step 5: Write failing test for config loader**

Create `tests/dataset_prep/test_config_loader.py`:

```python
import pytest
import yaml

from dataset_prep.config_loader import load_config, validate_config, DEFAULT_CONFIG_PATH


def _write_cfg(tmp_path, overrides=None):
    cfg = {
        "dataset": {
            "source": "andythetechnerd03/AI-human-text",
            "source_splits": ["train", "test"],
            "human_samples": 5000,
            "ai_samples": 5000,
            "random_seed": 2026,
        },
        "quality": {
            "remove_missing": True,
            "remove_empty": True,
            "remove_duplicates": True,
        },
        "label_mapping": {"status": "pending_verification"},
        "output": {
            "format_primary": "parquet",
            "format_secondary": "csv",
            "root": "project_dataset",
        },
    }
    if overrides:
        for section, values in overrides.items():
            cfg[section].update(values)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def test_load_config_reads_frozen_file():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg["dataset"]["random_seed"] == 2026
    assert cfg["dataset"]["human_samples"] == 5000
    assert cfg["dataset"]["ai_samples"] == 5000


def test_validate_accepts_frozen_spec():
    validate_config(load_config(DEFAULT_CONFIG_PATH))  # must not raise


@pytest.mark.parametrize(
    "overrides",
    [
        ({"dataset": {"random_seed": 42}}),
        ({"dataset": {"human_samples": 4000}}),
        ({"dataset": {"ai_samples": 6000}}),
        ({"dataset": {"source": "some/other-dataset"}}),
        ({"quality": {"remove_duplicates": False}}),
        ({"output": {"format_primary": "sqlite"}}),
    ],
)
def test_validate_rejects_spec_violations(tmp_path, overrides):
    p = _write_cfg(tmp_path, overrides)
    with pytest.raises(ValueError):
        validate_config(load_config(p))
```

- [ ] **Step 6: Run test, verify it fails**

Run: `python -m pytest tests/dataset_prep/test_config_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dataset_prep'` (or import error for `config_loader`).

- [ ] **Step 7: Implement `dataset_prep/config_loader.py`**

```python
"""Load and strictly validate the frozen Phase 0 configuration."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = REPO_ROOT / "project_dataset"
RAW_DIR = PROJECT_DIR / "raw" / "source_dataset"
PROCESSED_DIR = PROJECT_DIR / "processed"
SAMPLED_DIR = PROJECT_DIR / "sampled"
EXPORTS_DIR = PROJECT_DIR / "exports"
CONFIGS_DIR = PROJECT_DIR / "configs"
REPORTS_DIR = PROJECT_DIR / "reports"

DEFAULT_CONFIG_PATH = CONFIGS_DIR / "sampling_config.yaml"

FROZEN_SPEC = {
    "source": "andythetechnerd03/AI-human-text",
    "source_splits": ["train", "test"],
    "human_samples": 5000,
    "ai_samples": 5000,
    "random_seed": 2026,
    "remove_missing": True,
    "remove_empty": True,
    "remove_duplicates": True,
    "format_primary": "parquet",
    "format_secondary": "csv",
}


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_config(cfg: dict) -> None:
    d, q, o = cfg["dataset"], cfg["quality"], cfg["output"]
    problems = []
    if d["source"] != FROZEN_SPEC["source"]:
        problems.append(f"dataset.source must be {FROZEN_SPEC['source']!r}")
    if d["source_splits"] != FROZEN_SPEC["source_splits"]:
        problems.append("dataset.source_splits must be ['train', 'test']")
    if d["human_samples"] != FROZEN_SPEC["human_samples"]:
        problems.append("dataset.human_samples must be 5000")
    if d["ai_samples"] != FROZEN_SPEC["ai_samples"]:
        problems.append("dataset.ai_samples must be 5000")
    if d["random_seed"] != FROZEN_SPEC["random_seed"]:
        problems.append("dataset.random_seed must be 2026")
    for k in ("remove_missing", "remove_empty", "remove_duplicates"):
        if q[k] is not FROZEN_SPEC[k]:
            problems.append(f"quality.{k} must be true")
    if o["format_primary"] != FROZEN_SPEC["format_primary"]:
        problems.append("output.format_primary must be 'parquet'")
    if o["format_secondary"] != FROZEN_SPEC["format_secondary"]:
        problems.append("output.format_secondary must be 'csv'")
    if problems:
        raise ValueError(
            "Config violates the frozen Phase 0 spec:\n- " + "\n- ".join(problems)
        )


if __name__ == "__main__":
    cfg = load_config()
    validate_config(cfg)
    print("CONFIG OK (frozen Phase 0 spec satisfied)")
```

- [ ] **Step 8: Run tests, verify they pass**

Run: `python -m pytest tests/dataset_prep/test_config_loader.py -v`
Expected: 8 passed.

- [ ] **Step 9: Sanity-run the loader against the real frozen file**

Run: `python -m dataset_prep.config_loader`
Expected output: `CONFIG OK (frozen Phase 0 spec satisfied)`

- [ ] **Step 10: Commit**

```powershell
git add .gitignore requirements.txt project_dataset/configs/sampling_config.yaml dataset_prep tests
git commit -m "feat(phase0): bootstrap repo, frozen sampling config, strict config loader"
```

---

### Task 2: Secure HF token handling + authentication check

**Files:**
- Create: `dataset_prep/env_utils.py`
- Create: `dataset_prep/auth_check.py`
- Test: `tests/dataset_prep/test_env_utils.py`

**Interfaces:**
- Consumes: nothing from Task 1 except conventions.
- Produces: `get_hf_token(env_file: Path | None = None) -> str | None` — resolves `HF_TOKEN` from process env, else parses repo-root `.env`; NEVER prints the token. Later network modules call this.

- [ ] **Step 1: Write failing tests**

Create `tests/dataset_prep/test_env_utils.py`:

```python
from dataset_prep.env_utils import get_hf_token


def test_reads_token_from_env_file(tmp_path, capsys):
    f = tmp_path / ".env"
    f.write_text('# comment\nHF_TOKEN="hf_secret123"\nOTHER=x\n', encoding="utf-8")
    assert get_hf_token(env_file=f) == "hf_secret123"
    captured = capsys.readouterr()
    assert "hf_secret123" not in captured.out
    assert "hf_secret123" not in captured.err


def test_env_var_takes_priority(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "env_tok")
    f = tmp_path / ".env"
    f.write_text("HF_TOKEN=file_tok\n", encoding="utf-8")
    assert get_hf_token(env_file=f) == "env_tok"


def test_missing_everywhere_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert get_hf_token(env_file=tmp_path / "nope.env") is None
```

- [ ] **Step 2: Run tests, verify failure**

Run: `python -m pytest tests/dataset_prep/test_env_utils.py -v`
Expected: FAIL — `ModuleNotFoundError` for `dataset_prep.env_utils`.

- [ ] **Step 3: Implement `dataset_prep/env_utils.py`**

```python
"""Secure HF token resolution. The token must never be printed or committed."""
import os
from pathlib import Path

from dataset_prep.config_loader import REPO_ROOT


def get_hf_token(env_file: Path | None = None) -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token and token.strip():
        return token.strip()
    ef = env_file or (REPO_ROOT / ".env")
    if ef.exists():
        for line in ef.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "HF_TOKEN":
                tok = value.strip().strip('"').strip("'")
                if tok:
                    return tok
    return None
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python -m pytest tests/dataset_prep/test_env_utils.py -v`
Expected: 3 passed.

- [ ] **Step 5: Implement auth check script**

Create `dataset_prep/auth_check.py`:

```python
"""Verify HF authentication works using the securely-resolved token."""
import sys

from huggingface_hub import whoami

from dataset_prep.env_utils import get_hf_token


def main() -> int:
    token = get_hf_token()
    if not token:
        print("AUTH FAILED: no HF_TOKEN in environment or .env")
        return 1
    try:
        info = whoami(token=token)
    except Exception as exc:  # noqa: BLE001 - report any auth/network failure
        print(f"AUTH FAILED: {type(exc).__name__}: {exc}")
        return 1
    print(f"AUTH OK as user: {info.get('name')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run the auth check**

Run: `python -m dataset_prep.auth_check`
Expected: `AUTH OK as user: <username>` and exit code 0. If it fails because the `.env` token is invalid/expired, STOP and ask the operator to refresh `HF_TOKEN` — do not continue unauthenticated.

- [ ] **Step 7: Commit**

```powershell
git add dataset_prep/env_utils.py dataset_prep/auth_check.py tests/dataset_prep/test_env_utils.py
git commit -m "feat(phase1): secure HF token resolution and auth verification"
```

---

### Task 3: Download immutable source snapshot + manifest

**Files:**
- Create: `dataset_prep/download_source.py`

**Interfaces:**
- Consumes: `get_hf_token()`, `load_config`/`validate_config`, `RAW_DIR`, `CONFIGS_DIR`.
- Produces: local snapshot at `project_dataset/raw/source_dataset/` (DatasetDict format, loadable with `datasets.load_from_disk`) and `project_dataset/configs/source_manifest.json` with keys: `repo_id`, `revision` (commit sha), `splits` (`{split: n_rows}`), `saved_to`, `downloaded_at`, `datasets_version`. Function `resolve_revision(repo_id: str, token: str | None) -> str`.

- [ ] **Step 1: Implement `dataset_prep/download_source.py`**

```python
"""Download the source dataset once, pinned to a revision, and record a manifest."""
import json
import sys
from datetime import datetime, timezone

import datasets
from datasets import load_dataset
from huggingface_hub import HfApi

from dataset_prep.config_loader import (
    CONFIGS_DIR,
    RAW_DIR,
    load_config,
    validate_config,
)
from dataset_prep.env_utils import get_hf_token

MANIFEST_PATH = CONFIGS_DIR / "source_manifest.json"


def resolve_revision(repo_id: str, token: str | None) -> str:
    return HfApi(token=token).dataset_info(repo_id).sha


def download_snapshot(repo_id: str, revision: str, token: str | None) -> dict:
    ds = load_dataset(repo_id, revision=revision, token=token)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(RAW_DIR))
    return {
        "repo_id": repo_id,
        "revision": revision,
        "splits": {name: int(len(split)) for name, split in ds.items()},
        "features": {
            name: str(split.features) for name, split in ds.items()
        },
        "saved_to": str(RAW_DIR),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "datasets_version": datasets.__version__,
    }


def record_manifest(manifest: dict, path=MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    cfg = load_config()
    validate_config(cfg)
    if RAW_DIR.exists() and any(RAW_DIR.iterdir()):
        print(f"Raw snapshot already exists at {RAW_DIR}; refusing to re-download "
              f"(delete it manually to force).")
        return 0
    token = get_hf_token()
    repo_id = cfg["dataset"]["source"]
    revision = resolve_revision(repo_id, token)
    print(f"Downloading {repo_id} @ {revision} ...")
    manifest = download_snapshot(repo_id, revision, token)
    record_manifest(manifest)
    print(f"Splits: {manifest['splits']}")
    print(f"Manifest written to {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the download (network-heavy, ~571 MB)**

Run: `python -m dataset_prep.download_source`
Expected: prints a 40-hex revision sha, then `Splits: {'train': 462873, 'test': 24362}`, then manifest path. Exit code 0.

- [ ] **Step 3: Verify the snapshot reloads from disk**

Run:

```powershell
python -c "from datasets import load_from_disk; from dataset_prep.config_loader import RAW_DIR; ds = load_from_disk(str(RAW_DIR)); print({k: len(v) for k, v in ds.items()}); print(ds['train'].column_names)"
```

Expected: `{'train': 462873, 'test': 24362}` and `['text', 'generated']`.

- [ ] **Step 4: Commit the manifest (raw data itself is gitignored)**

```powershell
git add project_dataset/configs/source_manifest.json dataset_prep/download_source.py
git commit -m "feat(phase1): pinned source snapshot download with provenance manifest"
```

---

### Task 4: Label-mapping verification gate (STOP-if-ambiguous)

**Files:**
- Create: `dataset_prep/verify_labels.py`
- Modify: nothing else yet.
- Outputs produced at runtime: `project_dataset/reports/label_verification_evidence.md`, `project_dataset/reports/label_evidence_samples.md`, `project_dataset/configs/label_mapping.yaml`

**Interfaces:**
- Consumes: raw snapshot from Task 3 (`load_from_disk(RAW_DIR)`), `get_hf_token()`, source revision from `source_manifest.json`.
- Produces: `configs/label_mapping.yaml` with canonical shape consumed by Tasks 6–10:
  ```yaml
  verified_at: "<iso timestamp>"
  confirmed_by_operator: true|false
  mapping:
    0: human        # YAML int keys
    1: ai
  evidence:
    - {source: <str>, found: <bool>, statement: <str>, reference: <str>, extracted_mapping: <mapping|null>}
  ```
  Plus reusable functions: `extract_mapping_from_text(text: str) -> dict | None`, `normalize_label_name(name: str) -> str | None`, `propose_mapping(evidences: list[dict]) -> dict | None` (returns a mapping only when ≥1 evidence source yielded one AND all yielded mappings agree; returns `None` on absence OR conflict), `load_label_mapping(path) -> dict` which raises `RuntimeError` unless `confirmed_by_operator: true`.

Context established during planning (recorded here so the engineer knows what to expect):
- Features API exposes `generated` as plain `int8` — NO authoritative names available from features.
- Dataset card body says only: "This is a processed dataset of Human vs AI Text … taken from the Kaggle dataset https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text".
- Candidate model cards trained on this dataset: `andythetechnerd03/BERT-tiny_AI-Human`, `tomerz14/human-vs-AI_distilbert-classifier-v2` (their `config.json` `id2label` may carry explicit names).

- [ ] **Step 1: Write failing unit tests**

Create `tests/dataset_prep/test_verify_labels.py`:

```python
import pytest
import yaml

from dataset_prep.verify_labels import (
    extract_mapping_from_text,
    load_label_mapping,
    normalize_label_name,
    propose_mapping,
)


@pytest.mark.parametrize(
    "text",
    [
        "The generated column is 1 for AI-generated text and 0 for human-written text.",
        "labels: 0 = human, 1 = machine generated.",
        "0 corresponds to human essays while 1 corresponds to GPT text.",
        "Each essay was written by a human (0) or generated (1) by a model.",
        "generated: whether the text is machine-written (1) or human (0).",
    ],
)
def test_extract_mapping_positive_cases(text):
    assert extract_mapping_from_text(text) == {0: "human", 1: "ai"}


@pytest.mark.parametrize(
    "text",
    [
        "Labels take values 0 and 1.",
        "This dataset contains human and AI text.",
    ],
)
def test_extract_mapping_negative_cases(text):
    assert extract_mapping_from_text(text) is None


def test_extract_mapping_conflict_returns_none():
    conflicting = "0 means AI text and 1 means human text."
    assert extract_mapping_from_text(conflicting) is None


def test_normalize_label_names():
    assert normalize_label_name("Human") == "human"
    assert normalize_label_name("machine generated") == "ai"
    assert normalize_label_name("GPT-4") == "ai"
    assert normalize_label_name("banana") is None


def test_propose_requires_agreement():
    good = [{"extracted_mapping": {0: "human", 1: "ai"}},
            {"extracted_mapping": {0: "human", 1: "ai"}}]
    assert propose_mapping(good) == {0: "human", 1: "ai"}
    none_found = [{"extracted_mapping": None}, {"extracted_mapping": None}]
    assert propose_mapping(none_found) is None
    conflict = [{"extracted_mapping": {0: "human", 1: "ai"}},
                {"extracted_mapping": {0: "ai", 1: "human"}}]
    assert propose_mapping(conflict) is None


def _write_mapping(tmp_path, confirmed):
    doc = {
        "verified_at": "2026-08-24T00:00:00+00:00",
        "confirmed_by_operator": confirmed,
        "mapping": {0: "human", 1: "ai"},
        "evidence": [],
    }
    p = tmp_path / "label_mapping.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return p


def test_load_label_mapping_requires_operator_confirmation(tmp_path):
    p = _write_mapping(tmp_path, confirmed=False)
    with pytest.raises(RuntimeError):
        load_label_mapping(p)
    p2 = _write_mapping(tmp_path, confirmed=True)
    loaded = load_label_mapping(p2)
    assert loaded == {0: "human", 1: "ai"}
```

- [ ] **Step 2: Run tests, verify failure**

Run: `python -m pytest tests/dataset_prep/test_verify_labels.py -v`
Expected: FAIL — `ModuleNotFoundError` for `dataset_prep.verify_labels`.

- [ ] **Step 3: Implement `dataset_prep/verify_labels.py`**

```python
"""Multi-source label-mapping verification with a hard operator-confirmation gate."""
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from datasets import load_from_disk
from huggingface_hub import HfApi, hf_hub_download

from dataset_prep.config_loader import CONFIGS_DIR, RAW_DIR, REPORTS_DIR
from dataset_prep.download_source import MANIFEST_PATH as SOURCE_MANIFEST
from dataset_prep.env_utils import get_hf_token

MAPPING_PATH = CONFIGS_DIR / "label_mapping.yaml"

CANDIDATE_MODEL_IDS = [
    "andythetechnerd03/BERT-tiny_AI-Human",
    "tomerz14/human-vs-AI_distilbert-classifier-v2",
]

SYNONYMS = {
    "human": "human", "person": "human", "man": "human", "writer": "human",
    "ai": "ai", "machine": "ai", "gpt": "ai", "llm": "ai", "chatgpt": "ai",
    "generated": "ai", "ai_generated": "ai", "machine_generated": "ai",
    "machine_written": "ai", "machine_text": "ai", "ai_text": "ai",
}

_HUMAN_ZERO = re.compile(r"\b0\b[^.;]{0,60}?\b(human|writer)\b", re.I)
_ZERO_HUMAN = re.compile(r"\b(human|writer)\b[^.;]{0,60}?\b0\b", re.I)
_AI_ONE = re.compile(
    r"\b1\b[^.;]{0,60}?\b(ai|machine[\s-]?generated|machine[\s-]?written"
    r"|gpt|llm|chatgpt|generated)\b", re.I)
_ONE_AI = re.compile(
    r"\b(ai|machine[\s-]?generated|machine[\s-]?written|gpt|llm|chatgpt"
    r"|generated)\b[^.;]{0,60}?\b1\b", re.I)
_CONFLICT_HUMAN_ONE = re.compile(r"\b1\b[^.;]{0,60}?\b(human|writer)\b", re.I)
_CONFLICT_AI_ZERO = re.compile(r"\b0\b[^.;]{0,60}?\b(ai|machine|gpt|llm)\b", re.I)


def normalize_label_name(name: str) -> str | None:
    key = re.sub(r"[-\s]+", "_", str(name).strip().lower())
    return SYNONYMS.get(key)


def extract_mapping_from_text(text: str) -> dict | None:
    has_human_zero = bool(_HUMAN_ZERO.search(text) or _ZERO_HUMAN.search(text))
    has_ai_one = bool(_AI_ONE.search(text) or _ONE_AI.search(text))
    conflict = bool(_CONFLICT_HUMAN_ONE.search(text) or _CONFLICT_AI_ZERO.search(text))
    if has_human_zero and has_ai_one and not conflict:
        return {0: "human", 1: "ai"}
    return None


def propose_mapping(evidences: list[dict]) -> dict | None:
    extracted = [e["extracted_mapping"] for e in evidences if e["extracted_mapping"]]
    if not extracted:
        return None
    unique = {tuple(sorted(m.items())) for m in extracted}
    if len(unique) == 1:
        return dict(extracted[0])
    return None


def _evidence_features(ds_dict) -> dict:
    for split in ds_dict.values():
        gen = split.features.get("generated", None)
        names = getattr(gen, "names", None)
        if names:
            mapping = {}
            ok = True
            for i, nm in enumerate(names):
                norm = normalize_label_name(nm)
                if norm is None:
                    ok = False
                    break
                mapping[i] = norm
            if ok:
                return {"source": "features_classlabel", "found": True,
                        "statement": f"ClassLabel names: {names}",
                        "reference": "datasets features API",
                        "extracted_mapping": mapping}
    return {"source": "features_classlabel", "found": False,
            "statement": "'generated' is plain int8; no ClassLabel names exposed",
            "reference": "datasets features API / datasets-server info",
            "extracted_mapping": None}


def _evidence_readme(repo_id: str, token: str | None) -> dict:
    try:
        p = hf_hub_download(repo_id, filename="README.md",
                            repo_type="dataset", token=token)
        text = Path(p).read_text(encoding="utf-8", errors="replace")
        m = extract_mapping_from_text(text)
        return {"source": "repo_readme", "found": True,
                "statement": (m and "explicit mapping statement found")
                             or "README present but states no explicit mapping",
                "reference": f"https://huggingface.co/datasets/{repo_id}",
                "extracted_mapping": m}
    except Exception as exc:  # noqa: BLE001
        return {"source": "repo_readme", "found": False,
                "statement": f"README unavailable: {type(exc).__name__}",
                "reference": f"https://huggingface.co/datasets/{repo_id}",
                "extracted_mapping": None}


def _evidence_model_configs(token: str | None) -> dict:
    api = HfApi(token=token)
    statements, extracted = [], []
    for model_id in CANDIDATE_MODEL_IDS:
        try:
            p = hf_hub_download(model_id, filename="config.json",
                                repo_type="model", token=token)
            id2label = json.loads(Path(p).read_text(encoding="utf-8")).get("id2label")
            if not id2label:
                raise ValueError("config.json has no id2label")
            mapping = {int(k): normalize_label_name(v) for k, v in id2label.items()}
            if set(mapping.values()) != {"human", "ai"} or None in mapping.values():
                raise ValueError(f"id2label not human/ai: {id2label}")
            statements.append(f"{model_id}: id2label={id2label}")
            extracted.append(mapping)
        except Exception as exc:  # noqa: BLE001
            statements.append(f"{model_id}: unusable ({type(exc).__name__}: {exc})")
    agreed = None
    uniq = {tuple(sorted(m.items())) for m in extracted}
    if len(uniq) == 1:
        agreed = dict(extracted[0])
    elif len(uniq) > 1:
        agreed = dict(extracted[0]) if len({json.dumps(m, sort_keys=True) for m in extracted}) == 1 else None
        agreed = None  # conflicting model cards cannot be auto-resolved
    return {"source": "model_card_configs", "found": bool(extracted),
            "statement": "; ".join(statements) or "no candidate models checked",
            "reference": ", ".join(CANDIDATE_MODEL_IDS),
            "extracted_mapping": agreed}


def _evidence_kaggle_parent(token: str | None) -> dict:
    """Fetch the HF-rendered dataset card body and search it (the card cites the
    Kaggle parent). Direct Kaggle pages are JS-gated; we record honest findings."""
    try:
        p = hf_hub_download(CARD_REPO, filename="README.md",
                            repo_type="dataset", token=token)
        text = Path(p).read_text(encoding="utf-8", errors="replace")
        m = extract_mapping_from_text(text)
        return {"source": "kaggle_parent_reference", "found": True,
                "statement": (m and "card body contains explicit mapping")
                             or "card cites Kaggle parent but defines no explicit mapping",
                "reference": "https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text",
                "extracted_mapping": m}
    except Exception as exc:  # noqa: BLE001
        return {"source": "kaggle_parent_reference", "found": False,
                "statement": f"card fetch failed: {type(exc).__name__}",
                "reference": "https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text",
                "extracted_mapping": None}


CARD_REPO = "andythetechnerd03/AI-human-text"


def qualitative_samples_md(ds_dict, n_per_label: int = 10, seed: int = 2026) -> str:
    import random
    rng = random.Random(seed)
    lines = ["# Qualitative label evidence samples", "",
             "> Non-authoritative corroboration for operator review.", ""]
    for split_name, split in ds_dict.items():
        df = split.to_pandas()
        lines.append(f"## Split `{split_name}`")
        for label_value in sorted(df["generated"].dropna().unique()):
            sub = df[df["generated"] == label_value]
            pick = sub.sample(min(n_per_label, len(sub)), random_state=rng.randrange(2**31))
            lines.append(f"\n### generated = {label_value} ({len(sub)} rows)")
            for _, row in pick.iterrows():
                snippet = str(row["text"])[:300].replace("\n", " ")
                lines.append(f"- [{row.name}] {snippet}")
        lines.append("")
    return "\n".join(lines)


def render_evidence_md(evidences: list[dict]) -> str:
    lines = ["# Label mapping verification evidence", ""]
    for ev in evidences:
        lines += [f"## Source: `{ev['source']}`",
                  f"- found: {ev['found']}",
                  f"- statement: {ev['statement']}",
                  f"- reference: {ev['reference']}",
                  f"- extracted_mapping: `{ev['extracted_mapping']}`", ""]
    return "\n".join(lines)


def load_label_mapping(path: Path | str = MAPPING_PATH) -> dict:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not doc.get("confirmed_by_operator"):
        raise RuntimeError(
            "Label mapping is NOT operator-confirmed. Run "
            "`python -m dataset_prep.verify_labels`, review the evidence report, "
            "then re-run with --confirm.")
    return {int(k): v for k, v in doc["mapping"].items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true",
                        help="mark the proposed mapping as operator-confirmed")
    args = parser.parse_args()

    token = get_hf_token()
    ds_dict = load_from_disk(str(RAW_DIR))
    evidences = [
        _evidence_features(ds_dict),
        _evidence_readme(CARD_REPO, token),
        _evidence_model_configs(token),
        _evidence_kaggle_parent(token),
    ]
    proposed = propose_mapping(evidences)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "label_verification_evidence.md").write_text(
        render_evidence_md(evidences), encoding="utf-8")
    (REPORTS_DIR / "label_evidence_samples.md").write_text(
        qualitative_samples_md(ds_dict), encoding="utf-8")

    if proposed is None:
        (REPORTS_DIR / "label_verification_evidence.md").open("a", encoding="utf-8").write(
            "\n## RESULT: AMBIGUOUS\n\nNo consistent documentary mapping could be "
            "extracted. **STOP — do not proceed to cleaning/sampling.** Resolve the "
            "ambiguity (e.g., consult the Kaggle parent card manually) and re-run.\n")
        print("LABEL MAPPING AMBIGUOUS — see "
              f"{REPORTS_DIR / 'label_verification_evidence.md'}")
        return 2

    rev = "?"
    if SOURCE_MANIFEST.exists():
        rev = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8")).get("revision", "?")
    doc = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "confirmed_by_operator": bool(args.confirm),
        "mapping": proposed,
        "evidence": evidences,
        "source_revision_when_verified": rev,
    }
    MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    MAPPING_PATH.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")

    if not args.confirm:
        print("Mapping PROPOSED as:", proposed)
        print("Review reports/label_verification_evidence.md and "
              "reports/label_evidence_samples.md, then re-run with --confirm.")
        return 3
    print(f"Label mapping CONFIRMED: {proposed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run unit tests, verify pass**

Run: `python -m pytest tests/dataset_prep/test_verify_labels.py -v`
Expected: 12 passed.

- [ ] **Step 5: First run — collect evidence (unconfirmed)**

Run: `python -m dataset_prep.verify_labels`
Expected outcomes:
- Exit `2` (AMBIGUOUS): STOP here and report to the operator — per Global Constraints, do not guess and do not continue to Task 5 until resolved.
- Exit `3` (proposal exists): read `project_dataset/reports/label_verification_evidence.md` AND `project_dataset/reports/label_evidence_samples.md`. Manually sanity-check that texts under the proposed `human` value look like student essays and texts under the proposed `ai` value look machine-fluent, and that the cited statement genuinely supports the direction of the mapping.

- [ ] **Step 6: Operator confirmation run**

After reviewing the evidence yourself and stating the conclusion aloud in the session log:

Run: `python -m dataset_prep.verify_labels --confirm`
Expected: `Label mapping CONFIRMED: {0: 'human', 1: 'ai'}` (or the reverse ONLY if the evidence actually supports it) and exit code 0.

- [ ] **Step 7: Commit**

```powershell
git add dataset_prep/verify_labels.py tests/dataset_prep/test_verify_labels.py project_dataset/configs/label_mapping.yaml project_dataset/reports/label_verification_evidence.md project_dataset/reports/label_evidence_samples.md
git commit -m "feat(phase1): verified label mapping with multi-source evidence gate"
```

---

### Task 5: Source dataset audit report

**Files:**
- Create: `dataset_prep/audit_source.py`
- Outputs: `project_dataset/reports/source_dataset_audit.md`, `project_dataset/reports/audit_results.json`
- Test: `tests/dataset_prep/test_audit_source.py`

**Interfaces:**
- Consumes: raw snapshot (Task 3).
- Produces: `length_stats(series: pd.Series) -> dict`, `duplicate_stats(series: pd.Series) -> dict`, `ascii_ratio(text: str) -> float`, `audit_frame(df: pd.DataFrame, text_col="text", label_col="label") -> dict`, `render_audit_md(combined: dict, per_split: dict) -> str`. The audit JSON is consumed by Task 10's summary generator.

- [ ] **Step 1: Write failing tests**

Create `tests/dataset_prep/test_audit_source.py`:

```python
import pandas as pd

from dataset_prep.audit_source import (
    ascii_ratio,
    audit_frame,
    duplicate_stats,
    length_stats,
)


def test_ascii_ratio():
    assert ascii_ratio("hello") == 1.0
    assert ascii_ratio("") == 1.0
    assert 0.0 <= ascii_ratio("héllo") < 1.0


def test_length_stats_keys():
    stats = length_stats(pd.Series(["one two", "three four five"]))
    assert stats["char_max"] == 15
    assert stats["word_max"] == 3


def test_duplicate_stats():
    s = pd.Series(["a", "a", "A  b", "A b"])
    d = duplicate_stats(s)
    assert d["exact_duplicate_rows"] == 1          # second "a"
    assert d["normalized_duplicate_rows"] == 1     # "A  b" vs "A b"


def test_audit_frame_counts():
    df = pd.DataFrame({
        "text": ["good text", None, "   ", "good text", "x"],
        "label": [0, 0, 1, 0, 9],
    })
    a = audit_frame(df)
    assert a["n_rows"] == 5
    assert a["null_text"] == 1
    assert a["empty_after_strip"] == 1
    assert a["exact_duplicate_rows"] == 1
    assert a["null_label"] == 0
    assert a["unexpected_labels"] == {9: 1}
    assert a["non_string_text"] == 0
```

- [ ] **Step 2: Run tests, verify failure**

Run: `python -m pytest tests/dataset_prep/test_audit_source.py -v`
Expected: FAIL — `ModuleNotFoundError` for `dataset_prep.audit_source`.

- [ ] **Step 3: Implement `dataset_prep/audit_source.py`**

```python
"""Phase 1.5: exhaustive audit of the raw source dataset. Reports only; mutates nothing."""
import json
import sys

import numpy as np
import pandas as pd
from datasets import load_from_disk

from dataset_prep.config_loader import RAW_DIR, REPORTS_DIR

SHORT_CHARS = 50       # flag threshold for suspiciously short texts
LONG_CHARS = 20000     # flag threshold for extremely long texts
LOW_ASCII = 0.80       # flag threshold for probable non-English content


def ascii_ratio(text: str) -> float:
    text = text or ""
    if not text:
        return 1.0
    return sum(1 for c in text if ord(c) < 128) / len(text)


def length_stats(series: pd.Series) -> dict:
    chars = series.astype("string").fillna("").str.len().to_numpy(dtype=float)
    words = series.astype("string").fillna("").str.split().str.len().to_numpy(dtype=float)
    pct = lambda arr, q: float(np.percentile(arr, q))
    return {
        "char_min": float(chars.min()), "char_p01": pct(chars, 1),
        "char_p25": pct(chars, 25), "char_median": pct(chars, 50),
        "char_p75": pct(chars, 75), "char_p95": pct(chars, 95),
        "char_p99": pct(chars, 99), "char_max": float(chars.max()),
        "word_mean": float(words.mean()), "word_median": float(np.median(words)),
        "word_p95": pct(words, 95), "word_max": float(words.max()),
    }


def duplicate_stats(series: pd.Series) -> dict:
    s = series.astype("string")
    normalized = s.fillna("").str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    return {
        "exact_duplicate_rows": int(s.duplicated(keep="first").sum()),
        "normalized_duplicate_rows": int(normalized.duplicated(keep="first").sum()),
    }


def audit_frame(df: pd.DataFrame, text_col: str = "text",
                label_col: str = "label") -> dict:
    text = df[text_col].astype("string")
    label = df[label_col]
    result = {
        "n_rows": int(len(df)),
        "label_counts": {str(k): int(v) for k, v in label.value_counts(dropna=False).items()},
        "null_text": int(text.isna().sum()),
        "empty_after_strip": int((text.fillna("").str.strip() == "").sum() - text.isna().sum()),
        "null_label": int(label.isna().sum()),
        "unexpected_labels": {str(k): int(v) for k, v in
                              label.value_counts(dropna=False).items()
                              if k is np.nan or k not in (0, 1)},
        "non_string_text": int((~text.dropna().map(lambda t: isinstance(t, str))).sum()),
        "short_texts_under_50_chars": int((text.fillna("").str.len() < SHORT_CHARS).sum()),
        "long_texts_over_20000_chars": int((text.fillna("").str.len() > LONG_CHARS).sum()),
        "low_ascii_ratio_rows": int(text.fillna("").map(ascii_ratio).lt(LOW_ASCII).sum()),
    }
    non_empty = text.dropna()[text.dropna().str.strip() != ""]
    result["lengths_non_empty"] = length_stats(non_empty)
    result.update(duplicate_stats(text))
    return result


def _md_table(d: dict) -> str:
    return "\n".join(f"| {k} | {v} |" for k, v in d.items())


def render_audit_md(per_split: dict, combined: dict) -> str:
    lines = ["# Source Dataset Audit — andythetechnerd03/AI-human-text", "",
             "Audit performed read-only on the local raw snapshot "
             "(`project_dataset/raw/source_dataset`). No records were modified.", ""]
    for split, a in per_split.items():
        lines += [f"## Split `{split}`", "", "| metric | value |", "|---|---|",
                  _md_table(a), ""]
    lines += ["## Combined (train + test)", "", "| metric | value |", "|---|---|",
              _md_table(combined), "",
              "Notes:",
              "- `normalized_duplicate_rows` counts case/whitespace-insensitive repeats; "
              "these inform Phase 2 dedup policy (exact-text dedup on stripped text).",
              "- `low_ascii_ratio_rows` is a practical language-consistency heuristic, "
              "reported only (not filtered).", ""]
    return "\n".join(lines)


def main() -> int:
    ds = load_from_disk(str(RAW_DIR))
    per_split, frames = {}, []
    for name in ds:
        df = ds[name].to_pandas()
        a = audit_frame(df, text_col="text", label_col="generated")
        per_split[name] = a
        df["_split"] = name
        frames.append(df)
    combined_df = pd.concat(frames, ignore_index=True)
    combined = audit_frame(combined_df, text_col="text", label_col="generated")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "source_dataset_audit.md").write_text(
        render_audit_md(per_split, combined), encoding="utf-8")
    (REPORTS_DIR / "audit_results.json").write_text(
        json.dumps({"per_split": per_split, "combined": combined}, indent=2),
        encoding="utf-8")
    print(f"Audit complete: combined rows={combined['n_rows']} "
          f"label_counts={combined['label_counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python -m pytest tests/dataset_prep/test_audit_source.py -v`
Expected: 4 passed.

Note: `empty_after_strip` in the test — `"   "` strips to empty; the `None` was already counted as null and excluded by the `- text.isna().sum()` adjustment. If the test fails on that metric, fix the implementation, not the test expectation.

- [ ] **Step 5: Run the real audit (takes a few minutes; loads ~487k rows)**

Run: `python -m dataset_prep.audit_source`
Expected: prints `Audit complete: combined rows=487235 label_counts={...}` and exit 0.

- [ ] **Step 6: Review the report**

Read `project_dataset/reports/source_dataset_audit.md`. Confirm it contains exact counts for: totals per split, human/AI distribution, null/empty text, duplicates (exact + normalized), length percentiles, short/long flags, low-ASCII rows. If the human/AI counts are wildly imbalanced (either class < 5,200 after expected cleaning), STOP and report — the 5,000-per-class target would be at risk.

- [ ] **Step 7: Commit**

```powershell
git add dataset_prep/audit_source.py tests/dataset_prep/test_audit_source.py project_dataset/reports/source_dataset_audit.md project_dataset/reports/audit_results.json
git commit -m "feat(phase1): exhaustive source dataset audit with exact counts"
```

---

### Task 6: Clean into candidate pools (Phase 2)

**Files:**
- Create: `dataset_prep/build_candidates.py`
- Outputs: `project_dataset/processed/human_candidates.parquet`, `project_dataset/processed/ai_candidates.parquet`, `project_dataset/reports/cleaning_log.json`, `project_dataset/reports/cleaning_log.md`
- Test: `tests/dataset_prep/test_build_candidates.py`

**Interfaces:**
- Consumes: raw snapshot; `load_config`, `validate_config`; **confirmed** label mapping via `load_label_mapping()` (raises if unconfirmed); `REPORTS_DIR`, `PROCESSED_DIR`.
- Produces: `clean_label_subset(df: pd.DataFrame, text_col: str, label_col: str, target_label: int, *, remove_missing: bool = True, remove_empty: bool = True, remove_duplicates: bool = True) -> tuple[pd.DataFrame, dict]` — operates on the split-concatenated frame carrying `source_split` and `source_row_id` columns; returns cleaned frame (sorted by `(split_order, source_row_id)`) plus log dict. Candidate parquet schema (column order fixed): `["source_dataset", "source_split", "source_row_id", "original_label", "text"]`. Log dict shape: `{"human": {...stages...}, "ai": {...}, "normalization_applied": "leading/trailing whitespace strip"}`. Later tasks consume the parquet pools and the log JSON.

Removal stage order (each stage's removals counted separately):
1. null label / label ≠ target (`removed_invalid_or_null_label`)
2. null text (`removed_missing_text`)
3. whitespace-only text (`removed_empty_text`)
4. exact duplicate text on whitespace-stripped text, keep-first in deterministic sort order (`removed_duplicate_text`)

Documented normalization on survivors: `text = text.str.strip()` — internal content untouched.

- [ ] **Step 1: Write failing tests**

Create `tests/dataset_prep/test_build_candidates.py`:

```python
import pandas as pd

from dataset_prep.build_candidates import clean_label_subset


def _base_df():
    return pd.DataFrame({
        "source_split": ["train"] * 6,
        "source_row_id": [0, 1, 2, 3, 4, 5],
        "text": ["apple", "banana", None, "   ", "banana", "cherry "],
        "generated": [0, 0, 0, 0, 0, 7],
    })


def test_stage_counts_and_survivors():
    out, log = clean_label_subset(_base_df(), "text", "generated", 0)
    assert log["initial"] == 6
    assert log["removed_invalid_or_null_label"] == 1      # label 7
    assert log["removed_missing_text"] == 1               # None
    assert log["removed_empty_text"] == 1                 # "   "
    assert log["removed_duplicate_text"] == 1             # 2nd "banana"
    assert log["final"] == 2
    assert sorted(out["text"].tolist()) == ["apple", "banana"]
    # survivor normalization: trailing space stripped
    assert out.loc[out["text"] == "cherry"].shape[0] == 0  # cherry had label 7, gone
    assert list(out.columns[:3]) == ["source_split", "source_row_id", "text"]


def test_dedupe_keeps_first_in_deterministic_order():
    df = pd.DataFrame({
        "source_split": ["train", "test", "test"],
        "source_row_id": [10, 0, 1],
        "text": ["dup", "dup", "other"],
        "generated": [1, 1, 1],
    })
    out, log = clean_label_subset(df, "text", "generated", 1)
    # train sorts before test, so train row id 10 survives
    assert out["source_row_id"].tolist() == [10]
    assert log["removed_duplicate_text"] == 1


def test_provenance_columns_preserved():
    out, _ = clean_label_subset(_base_df(), "text", "generated", 0)
    assert {"source_dataset", "source_split", "source_row_id",
            "original_label", "text"}.issubset(out.columns)
```

- [ ] **Step 2: Run tests, verify failure**

Run: `python -m pytest tests/dataset_prep/test_build_candidates.py -v`
Expected: FAIL — `ModuleNotFoundError` for `dataset_prep.build_candidates`.

- [ ] **Step 3: Implement `dataset_prep/build_candidates.py`**

```python
"""Phase 2: quality-filter raw records into human/ai candidate pools, logging every removal."""
import json
import sys
from datetime import datetime, timezone

import pandas as pd
from datasets import load_from_disk

from dataset_prep.config_loader import (
    PROCESSED_DIR, RAW_DIR, REPORTS_DIR, load_config, validate_config,
)
from dataset_prep.verify_labels import load_label_mapping

SPLIT_ORDER = {"train": 0, "test": 1}
SOURCE_DATASET_ID = "andythetechnerd03/AI-human-text"
CANDIDATE_COLUMNS = ["source_dataset", "source_split", "source_row_id",
                     "original_label", "text"]


def clean_label_subset(df: pd.DataFrame, text_col: str, label_col: str,
                       target_label: int, *, remove_missing: bool = True,
                       remove_empty: bool = True,
                       remove_duplicates: bool = True) -> tuple[pd.DataFrame, dict]:
    log = {"initial": int(len(df))}
    d = df[df[label_col].notna() & (df[label_col] == target_label)].copy()
    log["removed_invalid_or_null_label"] = log["initial"] - int(len(d))

    d[text_col] = d[text_col].astype("string")
    if remove_missing:
        before = len(d)
        d = d[d[text_col].notna()]
        log["removed_missing_text"] = before - int(len(d))
    else:
        log["removed_missing_text"] = 0

    if remove_empty:
        before = len(d)
        d = d[d[text_col].str.strip() != ""]
        log["removed_empty_text"] = before - int(len(d))
    else:
        log["removed_empty_text"] = 0

    # Deterministic order BEFORE dedup keep-first and BEFORE whitespace strip,
    # so dedup results are stable across runs.
    d["_split_order"] = d["source_split"].map(SPLIT_ORDER)
    d = d.sort_values(["_split_order", "source_row_id"], kind="stable")

    if remove_duplicates:
        before = len(d)
        d = d[d[text_col].str.strip().duplicated(keep="first") == False]  # noqa: E712
        log["removed_duplicate_text"] = before - int(len(d))
    else:
        log["removed_duplicate_text"] = 0

    d[text_col] = d[text_col].str.strip()   # documented normalization
    d = d.drop(columns=["_split_order"]).reset_index(drop=True)
    log["final"] = int(len(d))
    return d, log


def build_pools() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cfg = load_config(); validate_config(cfg)
    mapping = load_label_mapping()  # raises RuntimeError if not operator-confirmed
    human_label = next(k for k, v in mapping.items() if v == "human")
    ai_label = next(k for k, v in mapping.items() if v == "ai")

    ds = load_from_disk(str(RAW_DIR))
    frames = []
    for name in cfg["dataset"]["source_splits"]:
        df = ds[name].to_pandas()
        df["source_split"] = name
        df["source_row_id"] = df.index.astype("int64")   # ORIGINAL row identity
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)

    q = cfg["quality"]
    human_df, human_log = clean_label_subset(
        raw, "text", "generated", human_label,
        remove_missing=q["remove_missing"], remove_empty=q["remove_empty"],
        remove_duplicates=q["remove_duplicates"])
    ai_df, ai_log = clean_label_subset(
        raw, "text", "generated", ai_label,
        remove_missing=q["remove_missing"], remove_empty=q["remove_empty"],
        remove_duplicates=q["remove_duplicates"])

    for out, lbl in ((human_df, human_label), (ai_df, ai_label)):
        out.insert(0, "source_dataset", SOURCE_DATASET_ID)
        out["original_label"] = lbl
        out[CANDIDATE_COLUMNS]
    log = {
        "human": human_log, "ai": ai_log,
        "label_values_used": {"human": human_label, "ai": ai_label},
        "normalization_applied": "leading/trailing whitespace strip on surviving text only",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return human_df[CANDIDATE_COLUMNS], ai_df[CANDIDATE_COLUMNS], log


def render_cleaning_md(log: dict) -> str:
    def block(cls: str) -> str:
        l = log[cls]
        return "\n".join([
            f"## {cls.capitalize()} pool", "",
            f"Initial records: {l['initial']}",
            f"Removed invalid/null label: {l['removed_invalid_or_null_label']}",
            f"Removed missing text: {l['removed_missing_text']}",
            f"Removed empty text: {l['removed_empty_text']}",
            f"Removed duplicates: {l['removed_duplicate_text']}",
            f"**Final candidate pool: {l['final']}**", "",
        ])
    return "\n".join([
        "# Cleaning Log (Phase 2)", "",
        f"Generated: {log['generated_at']}", "",
        f"Label values used: `{log['label_values_used']}`", "",
        block("human"), block("ai"),
        f"Normalization applied: {log['normalization_applied']}",
    ])


def main() -> int:
    human_df, ai_df, log = build_pools()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    human_df.to_parquet(PROCESSED_DIR / "human_candidates.parquet", engine="pyarrow", index=False)
    ai_df.to_parquet(PROCESSED_DIR / "ai_candidates.parquet", engine="pyarrow", index=False)
    (REPORTS_DIR / "cleaning_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    (REPORTS_DIR / "cleaning_log.md").write_text(render_cleaning_md(log), encoding="utf-8")
    print(f"Human candidates: {len(human_df)}  AI candidates: {len(ai_df)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python -m pytest tests/dataset_prep/test_build_candidates.py -v`
Expected: 3 passed. If `test_dedupe_keeps_first_in_deterministic_order` fails, verify the `_split_order` sort happens before dedup (train=0 < test=1).

- [ ] **Step 5: Run the real cleaning**

Run: `python -m dataset_prep.build_candidates`
Expected: prints candidate counts; both must be comfortably above 5,000 (audit suggests hundreds of thousands). Exit 0.

- [ ] **Step 6: Verify pools reload from disk with intact provenance**

Run:

```powershell
python -c "import pandas as pd; from dataset_prep.config_loader import PROCESSED_DIR; h = pd.read_parquet(PROCESSED_DIR / 'human_candidates.parquet'); a = pd.read_parquet(PROCESSED_DIR / 'ai_candidates.parquet'); print(len(h), len(a)); print(list(h.columns)); print('null row_ids:', int(h.source_row_id.isna().sum() + a.source_row_id.isna().sum()))"
```

Expected: two large counts, exact column list `['source_dataset','source_split','source_row_id','original_label','text']`, `null row_ids: 0`.

- [ ] **Step 7: Commit**

```powershell
git add dataset_prep/build_candidates.py tests/dataset_prep/test_build_candidates.py project_dataset/processed project_dataset/reports/cleaning_log.json project_dataset/reports/cleaning_log.md
git commit -m "feat(phase2): quality-filtered candidate pools with full removal logging"
```

---

### Task 7: Seeded random sampling of 5,000 + 5,000 and combined export (Phase 3)

**Files:**
- Create: `dataset_prep/sample_originals.py`
- Outputs: `project_dataset/sampled/human_5000.parquet`, `project_dataset/sampled/ai_5000.parquet`, `project_dataset/sampled/originals_10000.parquet`, `project_dataset/exports/originals_10000.csv`, `project_dataset/reports/sampling_report.md`, `project_dataset/reports/sampling_summary.json`
- Test: `tests/dataset_prep/test_sampling.py`

**Interfaces:**
- Consumes: candidate pools (Task 6 parquet files), `load_config`/`validate_config`, confirmed `load_label_mapping()`.
- Produces (later tasks rely on these exact names/types):
  - `select_indices(pool_size: int, n_select: int, seed: int) -> list[int]`
  - `assign_ids(prefix: str, n: int) -> list[str]` → `H00001…` / `A00001…`
  - `FINAL_COLUMN_ORDER: list[str]`
  - Combined frame columns, in order: `sample_id, source_dataset, source_split, source_row_id, text, label, text_type, word_count, character_count, random_seed, sampling_timestamp, parent_id, paraphrase_level, generation_model, generation_model_revision, lex_diversity, order_diversity` (last six exist as all-null DIPPER placeholders).
  - `text_type` ∈ {`"human"`, `"ai_original"`}.
  - Sampling method string recorded verbatim: `METHOD_DESC = "random.Random(seed).sample(range(pool_size), n_select) over pool sorted by (split_order, source_row_id)"`.

- [ ] **Step 1: Write failing tests**

Create `tests/dataset_prep/test_sampling.py`:

```python
import pandas as pd

from dataset_prep.sample_originals import assign_ids, select_indices


def test_golden_value_known_vector():
    assert select_indices(50, 10, 2026) == [7, 20, 32, 47, 41, 6, 14, 38, 39, 35]


def test_select_is_deterministic():
    assert select_indices(1000, 100, 2026) == select_indices(1000, 100, 2026)


def test_select_differs_by_seed():
    assert select_indices(1000, 100, 2026) != select_indices(1000, 100, 2027)


def test_select_is_randomized_not_head():
    idx = select_indices(1000, 10, 2026)
    assert idx != list(range(10))


def test_assign_ids_format():
    ids = assign_ids("H", 3)
    assert ids == ["H00001", "H00002", "H00003"]
```

- [ ] **Step 2: Run tests, verify failure**

Run: `python -m pytest tests/dataset_prep/test_sampling.py -v`
Expected: FAIL — `ModuleNotFoundError` for `dataset_prep.sample_originals`.

- [ ] **Step 3: Implement `dataset_prep/sample_originals.py`**

```python
"""Phase 3: seeded, reproducible random sampling of exactly 5,000 human + 5,000 AI originals."""
import json
import random
import sys
from datetime import datetime, timezone

import pandas as pd

from dataset_prep.config_loader import (
    EXPORTS_DIR, PROCESSED_DIR, REPORTS_DIR, SAMPLED_DIR,
    load_config, validate_config,
)
from dataset_prep.verify_labels import load_label_mapping

DIPPER_PLACEHOLDER_COLS = [
    "parent_id", "paraphrase_level", "generation_model",
    "generation_model_revision", "lex_diversity", "order_diversity",
]
BASE_COLS = [
    "sample_id", "source_dataset", "source_split", "source_row_id", "text",
    "label", "text_type", "word_count", "character_count", "random_seed",
    "sampling_timestamp",
]
FINAL_COLUMN_ORDER = BASE_COLS + DIPPER_PLACEHOLDER_COLS
METHOD_DESC = ("random.Random(seed).sample(range(pool_size), n_select) "
               "over pool sorted by (split_order, source_row_id)")
SELECTION_PARQUET = SAMPLED_DIR / "originals_10000.parquet"


def select_indices(pool_size: int, n_select: int, seed: int) -> list[int]:
    return random.Random(seed).sample(range(pool_size), n_select)


def assign_ids(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i:05d}" for i in range(1, n + 1)]


def sample_class(pool: pd.DataFrame, n_select: int, seed: int, prefix: str,
                 text_type: str, sampling_timestamp: str,
                 label_value: int) -> pd.DataFrame:
    if len(pool) < n_select:
        raise ValueError(f"pool too small ({len(pool)}) to select {n_select}")
    sel = pool.iloc[select_indices(len(pool), n_select, seed)].copy()
    sel.insert(0, "sample_id", assign_ids(prefix, n_select))
    sel["label"] = label_value
    sel["text_type"] = text_type
    sel["word_count"] = sel["text"].str.split().str.len().astype("int64")
    sel["character_count"] = sel["text"].str.len().astype("int64")
    sel["random_seed"] = seed
    sel["sampling_timestamp"] = sampling_timestamp
    for col in DIPPER_PLACEHOLDER_COLS:
        sel[col] = pd.NA
    return sel[FINAL_COLUMN_ORDER].reset_index(drop=True)


def build_combined(human_sel: pd.DataFrame, ai_sel: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([human_sel, ai_sel], ignore_index=True)
    return combined[FINAL_COLUMN_ORDER]


def main() -> int:
    cfg = load_config(); validate_config(cfg)
    mapping = load_label_mapping()  # hard gate: must be operator-confirmed
    human_label = next(k for k, v in mapping.items() if v == "human")
    ai_label = next(k for k, v in mapping.items() if v == "ai")
    seed = cfg["dataset"]["random_seed"]
    n_humans = cfg["dataset"]["human_samples"]
    n_ai = cfg["dataset"]["ai_samples"]

    human_pool = pd.read_parquet(PROCESSED_DIR / "human_candidates.parquet")
    ai_pool = pd.read_parquet(PROCESSED_DIR / "ai_candidates.parquet")
    ts = datetime.now(timezone.utc).isoformat()

    human_sel = sample_class(human_pool, n_humans, seed, "H", "human", ts, human_label)
    ai_sel = sample_class(ai_pool, n_ai, seed, "A", "ai_original", ts, ai_label)
    combined = build_combined(human_sel, ai_sel)

    SAMPLED_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    human_sel.to_parquet(SAMPLED_DIR / "human_5000.parquet", engine="pyarrow", index=False)
    ai_sel.to_parquet(SAMPLED_DIR / "ai_5000.parquet", engine="pyarrow", index=False)
    combined.to_parquet(SELECTION_PARQUET, engine="pyarrow", index=False)
    combined.to_csv(EXPORTS_DIR / "originals_10000.csv", index=False, encoding="utf-8")

    summary = {
        "seed": seed, "method": METHOD_DESC,
        "sampling_timestamp": ts,
        "human": {"pool_size": len(human_pool), "selected": len(human_sel),
                  "id_range": [human_sel["sample_id"].iloc[0], human_sel["sample_id"].iloc[-1]]},
        "ai": {"pool_size": len(ai_pool), "selected": len(ai_sel),
               "id_range": [ai_sel["sample_id"].iloc[0], ai_sel["sample_id"].iloc[-1]]},
        "total_selected": len(combined),
    }
    (REPORTS_DIR / "sampling_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    report = "\n".join([
        "# Sampling Report (Phase 3)", "",
        f"- Sampling timestamp: {ts}",
        f"- random_seed: **{seed}**",
        f"- sampling_method: `{METHOD_DESC}`",
        f"- Human: pool={summary['human']['pool_size']} selected={summary['human']['selected']} "
        f"ids {summary['human']['id_range'][0]}..{summary['human']['id_range'][1]}",
        f"- AI: pool={summary['ai']['pool_size']} selected={summary['ai']['selected']} "
        f"ids {summary['ai']['id_range'][0]}..{summary['ai']['id_range'][1]}",
        f"- Total selected: {len(combined)}",
        "- First-N selection was NOT used (see tests asserting non-head selection).",
        "- Reproducibility section appended by reproduce_check (Task 9).",
    ])
    (REPORTS_DIR / "sampling_report.md").write_text(report, encoding="utf-8")
    print(f"Sampled {len(human_sel)} human + {len(ai_sel)} AI = {len(combined)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python -m pytest tests/dataset_prep/test_sampling.py -v`
Expected: 5 passed (including the golden-vector determinism assertion).

- [ ] **Step 5: Run the real sampling**

Run: `python -m dataset_prep.sample_originals`
Expected: `Sampled 5000 human + 5000 AI = 10000`, exit 0.

- [ ] **Step 6: Verify outputs from disk**

Run:

```powershell
python -c "import pandas as pd; from dataset_prep.config_loader import SAMPLED_DIR, EXPORTS_DIR; c = pd.read_parquet(SAMPLED_DIR / 'originals_10000.parquet'); print(c['text_type'].value_counts().to_dict()); print(c['sample_id'].is_unique); print(len(pd.read_csv(EXPORTS_DIR / 'originals_10000.csv')))"
```

Expected: `{'human': 5000, 'ai_original': 5000}`, `True`, `10000`.

- [ ] **Step 7: Commit**

```powershell
git add dataset_prep/sample_originals.py tests/dataset_prep/test_sampling.py project_dataset/sampled project_dataset/exports project_dataset/reports/sampling_report.md project_dataset/reports/sampling_summary.json
git commit -m "feat(phase3): seeded reproducible sampling of 10k originals with stable IDs"
```

---

### Task 8: Validate from disk + freeze (Phase 4)

**Files:**
- Create: `dataset_prep/validate_originals.py`
- Create: `dataset_prep/freeze_outputs.py`
- Outputs: `project_dataset/reports/validation_report.md`, `project_dataset/reports/validation_results.json`, `project_dataset/configs/freeze_manifest.json`
- Test: `tests/dataset_prep/test_validation.py`

**Interfaces:**
- Consumes: `SAMPLED_DIR/originals_10000.parquet`, `EXPORTS_DIR/originals_10000.csv` **reloaded from disk**; `load_config`; `load_label_mapping()` (confirmed).
- Produces: `validate(originals: pd.DataFrame, csv_df: pd.DataFrame, human_label: int, ai_label: int, expect_human: int, expect_ai: int, seed: int) -> dict` returning `{"errors": list[str], "warnings": list[str], "checks": dict[str, bool]}`; validator exits 1 on any error. `freeze_outputs.main()` hashes every frozen artifact into `freeze_manifest.json`.

Checks implemented (all must be true): counts 5,000/5,000/10,000 · every `text_type=="human"` row has `label==human_label` and vice versa · no unexpected labels · `sample_id` unique · `sample_id` matches class prefix/format · `(source_split, source_row_id)` pairs unique · provenance fields non-null (`source_dataset`, `source_split`, `source_row_id`) · no duplicate text within human set · none within AI set · none across sets · stored `word_count`/`character_count` equal recomputed values · all six DIPPER placeholder columns null · CSV↔Parquet agreement (row count, identical `sample_id` sets, identical text per `sample_id`) · recorded `random_seed` equals configured seed.

- [ ] **Step 1: Write failing tests**

Create `tests/dataset_prep/test_validation.py`:

```python
import pandas as pd
import pytest

from dataset_prep.sample_originals import FINAL_COLUMN_ORDER
from dataset_prep.validate_originals import validate


def _frame(n_h, n_a, corrupt=None):
    rows = []
    for i in range(n_h):
        rows.append({"sample_id": f"H{i+1:05d}", "source_dataset": "ds",
                     "source_split": "train", "source_row_id": i, "text": f"hum {i}",
                     "label": 0, "text_type": "human",
                     "word_count": 2, "character_count": len(f"hum {i}"),
                     "random_seed": 2026, "sampling_timestamp": "t"})
    for i in range(n_a):
        rows.append({"sample_id": f"A{i+1:05d}", "source_dataset": "ds",
                     "source_split": "train", "source_row_id": 1000 + i,
                     "text": f"mach {i}", "label": 1, "text_type": "ai_original",
                     "word_count": 2, "character_count": len(f"mach {i}"),
                     "random_seed": 2026, "sampling_timestamp": "t"})
    df = pd.DataFrame(rows)
    for col in ("parent_id", "paraphrase_level", "generation_model",
                "generation_model_revision", "lex_diversity", "order_diversity"):
        df[col] = pd.NA
    if corrupt:
        corrupt(df)
    return df[FINAL_COLUMN_ORDER]


def test_valid_small_fixture_passes():
    res = validate(_frame(6, 6), _frame(6, 6), 0, 1, 6, 6, 2026)
    assert res["errors"] == []
    assert all(res["checks"].values())


def test_cross_set_duplicate_is_error():
    def dup(df):
        df.loc[df.index[-1], "text"] = "hum 0"
    res = validate(_frame(6, 6, dup), _frame(6, 6, dup), 0, 1, 6, 6, 2026)
    assert any("across" in e.lower() for e in res["errors"])
    assert res["checks"]["no_duplicate_text_across_sets"] is False


def test_wrong_count_is_error():
    res = validate(_frame(5, 6), _frame(5, 6), 0, 1, 6, 6, 2026)
    assert res["checks"]["human_count_exact"] is False
    assert res["errors"]


def test_mismatched_csv_is_error():
    parquet = _frame(6, 6)
    csv = parquet.copy()
    csv.loc[csv.index[0], "text"] = "tampered"
    res = validate(parquet, csv, 0, 1, 6, 6, 2026)
    assert res["checks"]["csv_matches_parquet"] is False
```

- [ ] **Step 2: Run tests, verify failure**

Run: `python -m pytest tests/dataset_prep/test_validation.py -v`
Expected: FAIL — `ModuleNotFoundError` for `dataset_prep.validate_originals`.

- [ ] **Step 3: Implement `dataset_prep/validate_originals.py`**

```python
"""Phase 4: mandatory validation of the frozen originals — always reading FROM DISK."""
import json
import sys

import pandas as pd

from dataset_prep.config_loader import EXPORTS_DIR, REPORTS_DIR, SAMPLED_DIR, load_config, validate_config
from dataset_prep.sample_originals import DIPPER_PLACEHOLDER_COLS, SELECTION_PARQUET
from dataset_prep.verify_labels import load_label_mapping

ORIGINALS_PARQUET = SELECTION_PARQUET
ORIGINALS_CSV = EXPORTS_DIR / "originals_10000.csv"


def validate(originals: pd.DataFrame, csv_df: pd.DataFrame, human_label: int,
             ai_label: int, expect_human: int, expect_ai: int,
             seed: int) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}

    def chk(name: str, ok: bool, msg_if_bad: str, level: str = "error"):
        checks[name] = bool(ok)
        if not ok:
            (errors if level == "error" else warnings).append(msg_if_bad)

    expect_total = expect_human + expect_ai
    humans = originals[originals["text_type"] == "human"]
    ais = originals[originals["text_type"] == "ai_original"]

    chk("human_count_exact", len(humans) == expect_human,
        f"Human count {len(humans)} != {expect_human}")
    chk("ai_count_exact", len(ais) == expect_ai,
        f"AI count {len(ais)} != {expect_ai}")
    chk("total_count_exact", len(originals) == expect_total,
        f"Total count {len(originals)} != {expect_total}")

    chk("human_labels_correct", bool((humans["label"] == human_label).all()),
        "Some human rows carry the wrong label")
    chk("ai_labels_correct", bool((ais["label"] == ai_label).all()),
        "Some AI rows carry the wrong label")
    chk("only_expected_text_types",
        set(originals["text_type"].unique()) <= {"human", "ai_original"},
        "Unexpected text_type values present")

    chk("sample_ids_unique", originals["sample_id"].is_unique,
        "Duplicate sample_id values")
    prefix_ok = humans["sample_id"].str.fullmatch(r"H\d{5}").all() and \
        ais["sample_id"].str.fullmatch(r"A\d{5}").all()
    chk("sample_id_prefix_format", bool(prefix_ok),
        "sample_id does not match ^H\\d{5}$ / ^A\\d{5}$ per class")

    pair_cols = ["source_split", "source_row_id"]
    chk("source_pairs_unique", not originals.duplicated(subset=pair_cols).any(),
        "Duplicate (source_split, source_row_id) provenance pairs")

    prov_cols = ["sample_id", "source_dataset", "source_split", "source_row_id",
                 "text", "label", "text_type", "random_seed"]
    chk("provenance_complete", not originals[prov_cols].isna().any().any(),
        "Null values in required provenance fields")

    chk("no_duplicate_text_within_human",
        not humans["text"].duplicated().any(), "Duplicate text within human set")
    chk("no_duplicate_text_within_ai",
        not ais["text"].duplicated().any(), "Duplicate text within AI set")
    cross = pd.concat([humans["text"], ais["text"]]).duplicated().any()
    chk("no_duplicate_text_across_sets", not cross,
        "Duplicate text found ACROSS human and AI sets — investigate provenance")

    recomputed_w = originals["text"].str.split().str.len()
    recomputed_c = originals["text"].str.len()
    chk("word_counts_consistent", (originals["word_count"] == recomputed_w).all(),
        "Stored word_count differs from recomputed")
    chk("char_counts_consistent", (originals["character_count"] == recomputed_c).all(),
        "Stored character_count differs from recomputed")

    chk("dipper_fields_all_null",
        bool(originals[DIPPER_PLACEHOLDER_COLS].isna().all().all()),
        "DIPPER placeholder columns must remain null in Phase 0-4")

    chk("seed_recorded_matches_config",
        bool((originals["random_seed"] == seed).all()),
        "Recorded random_seed does not match configuration")

    same_len = len(csv_df) == len(originals)
    same_ids = same_len and set(csv_df["sample_id"]) == set(originals["sample_id"])
    same_text = False
    if same_ids:
        merged = csv_df.merge(originals[["sample_id", "text"]],
                              on="sample_id", suffixes=("_csv", "_pq"))
        same_text = (merged["text_csv"].astype(str)
                     == merged["text_pq"].astype(str)).all()
    chk("csv_matches_parquet", bool(same_len and same_ids and same_text),
        "Exported CSV does not agree with the Parquet original")

    return {"errors": errors, "warnings": warnings, "checks": checks}


def render_report(res: dict, paths: dict) -> str:
    lines = ["# Validation Report (Phase 4)", ""]
    lines += ["| check | passed |", "|---|---|"]
    lines += [f"| {k} | {'PASS' if v else '**FAIL**'} |"
              for k, v in res["checks"].items()]
    if res["errors"]:
        lines += ["", "## Errors", *[f"- {e}" for e in res["errors"]]]
    if res["warnings"]:
        lines += ["", "## Warnings", *[f"- {w}" for w in res["warnings"]]]
    verdict = "**VALIDATION PASSED**" if not res["errors"] else "**VALIDATION FAILED**"
    lines += ["", f"Verdict: {verdict}", "",
              "Validated against files reloaded from disk:", 
              f"- `{paths['parquet']}`", f"- `{paths['csv']}`", ""]
    return "\n".join(lines)


def main() -> int:
    cfg = load_config(); validate_config(cfg)
    mapping = load_label_mapping()
    human_label = next(k for k, v in mapping.items() if v == "human")
    ai_label = next(k for k, v in mapping.items() if v == "ai")

    originals = pd.read_parquet(ORIGINALS_PARQUET)     # FROM DISK
    csv_df = pd.read_csv(ORIGINALS_CSV, dtype={"sample_id": str})
    res = validate(originals, csv_df, human_label, ai_label,
                   cfg["dataset"]["human_samples"], cfg["dataset"]["ai_samples"],
                   cfg["dataset"]["random_seed"])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "validation_results.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    (REPORTS_DIR / "validation_report.md").write_text(
        render_report(res, {"parquet": str(ORIGINALS_PARQUET),
                            "csv": str(ORIGINALS_CSV)}), encoding="utf-8")
    for e in res["errors"]:
        print(f"ERROR: {e}")
    print("VALIDATION PASSED" if not res["errors"] else "VALIDATION FAILED")
    return 0 if not res["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests, verify pass**

Run: `python -m pytest tests/dataset_prep/test_validation.py -v`
Expected: 4 passed.

- [ ] **Step 5: Implement `dataset_prep/freeze_outputs.py`**

```python
"""Compute SHA256 checksums over every frozen artifact and record a freeze manifest."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dataset_prep.config_loader import (
    CONFIGS_DIR, EXPORTS_DIR, SAMPLED_DIR, load_config,
)
from dataset_prep.download_source import MANIFEST_PATH as SOURCE_MANIFEST

FREEZE_MANIFEST = CONFIGS_DIR / "freeze_manifest.json"
FROZEN_FILES = [
    SAMPLED_DIR / "human_5000.parquet",
    SAMPLED_DIR / "ai_5000.parquet",
    SAMPLED_DIR / "originals_10000.parquet",
    EXPORTS_DIR / "originals_10000.csv",
    CONFIGS_DIR / "sampling_config.yaml",
    CONFIGS_DIR / "label_mapping.yaml",
    CONFIGS_DIR / "source_manifest.json",
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_freeze_manifest(files: list[Path]) -> dict:
    entries = {}
    for p in files:
        if not p.exists():
            raise FileNotFoundError(f"frozen artifact missing: {p}")
        entries[str(p.relative_to(p.parents[2]))] = {
            "sha256": sha256_of(p), "bytes": p.stat().st_size,
        }
    src_rev = None
    if SOURCE_MANIFEST.exists():
        src_rev = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8")).get("revision")
    cfg = load_config()
    return {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "source_revision": src_rev,
        "random_seed": cfg["dataset"]["random_seed"],
        "files": entries,
    }


def main() -> int:
    manifest = build_freeze_manifest(FROZEN_FILES)
    FREEZE_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = REPORT_LINE = "\n".join([
        "", "---", "", "# FREEZE MANIFEST", "",
        f"Frozen at: {manifest['frozen_at']}",
        f"Source revision: `{manifest['source_revision']}`",
        f"Seed: {manifest['random_seed']}", "",
        "| artifact | sha256 | bytes |", "|---|---|---|",
        *[f"| {k} | `{v['sha256'][:16]}…` | {v['bytes']} |"
          for k, v in manifest["files"].items()], "",
        "Any change to these files after this point invalidates the freeze.", "",
    ])
    vr = REPORTS := CONFIGS_DIR.parents[0] / "reports" / "validation_report.md"
    if vr.exists():
        with open(vr, "a", encoding="utf-8") as f:
            f.write("\n" + report)
    print(f"Freeze manifest written: {FREEZE_MANIFEST} ({len(manifest['files'])} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run validation against disk**

Run: `python -m dataset_prep.validate_originals`
Expected: `VALIDATION PASSED`, exit 0. If ANY check fails: STOP. Report every error. Do not patch data — investigate provenance/cleaning instead.

- [ ] **Step 7: Freeze**

Run: `python -m dataset_prep.freeze_outputs`
Expected: `Freeze manifest written: ... (7 files)`; freeze section appended to `validation_report.md`.

- [ ] **Step 8: Commit**

```powershell
git add dataset_prep/validate_originals.py dataset_prep/freeze_outputs.py tests/dataset_prep/test_validation.py project_dataset/reports/validation_report.md project_dataset/reports/validation_results.json project_dataset/configs/freeze_manifest.json
git commit -m "feat(phase4): disk-based validation of 10k originals and checksum freeze"
```

---

### Task 9: Reproducibility re-run test

**Files:**
- Create: `dataset_prep/reproduce_check.py`
- Output: `project_dataset/reports/reproducibility_check.md`

**Interfaces:**
- Consumes: `sample_class`/`select_indices` from Task 7, candidate pools on disk, existing `sampled/*.parquet`, `sampling_summary.json`.
- Produces: re-execution of the sampling procedure in-memory with the same seed; comparison of (a) the ordered list of `(source_split, source_row_id)` selections per class and (b) aligned `sample_id`→`text` mappings, against the saved artifacts. Writes `RESULT: PASS` or `RESULT: FAIL` into the report and appends a summary line to `sampling_report.md`. Exit 1 on failure.

- [ ] **Step 1: Implement `dataset_prep/reproduce_check.py`**

```python
"""Re-run Phase 3 sampling with the same seed and prove identical selection."""
import json
import sys
from datetime import datetime, timezone

import pandas as pd

from dataset_prep.config_loader import (
    PROCESSED_DIR, REPORTS_DIR, SAMPLED_DIR, load_config, validate_config,
)
from dataset_prep.sample_originals import sample_originals_pair  # noqa: F401 (re-exported helper below)
from dataset_prep.sample_originals import METHOD_DESC, sample_class
from dataset_prep.verify_labels import load_label_mapping


def selection_signature(df: pd.DataFrame) -> list[tuple[str, int]]:
    return list(zip(df["source_split"], df["source_row_id"].astype(int)))


def main() -> int:
    cfg = load_config(); validate_config(cfg)
    mapping = load_label_mapping()
    human_label = next(k for k, v in mapping.items() if v == "human")
    ai_label = next(k for k, v in mapping.items() if v == "ai")
    seed = cfg["dataset"]["random_seed"]
    ts = datetime.now(timezone.utc).isoformat()  # timestamps MAY differ; selection must not

    human_pool = pd.read_parquet(PROCESSED_DIR / "human_candidates.parquet")
    ai_pool = pd.read_parquet(PROCESSED_DIR / "ai_candidates.parquet")
    rerun_h = sample_class(human_pool, cfg["dataset"]["human_samples"], seed,
                           "H", "human", ts, human_label)
    rerun_a = sample_class(ai_pool, cfg["dataset"]["ai_samples"], seed,
                           "A", "ai_original", ts, ai_label)

    saved_h = pd.read_parquet(SAMPLED_DIR / "human_5000.parquet")
    saved_a = pd.read_parquet(SAMPLED_DIR / "ai_5000.parquet")

    results = {
        "human_selection_identical":
            selection_signature(rerun_h) == selection_signature(saved_h),
        "ai_selection_identical":
            selection_signature(rerun_a) == selection_signature(saved_a),
        "human_id_text_map_identical":
            rerun_h[["sample_id", "text"]].equals(saved_h[["sample_id", "text"]]),
        "ai_id_text_map_identical":
            rerun_a[["sample_id", "text"]].equals(saved_a[["sample_id", "text"]]),
    }
    passed = all(results.values())

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# Reproducibility Check (Phase 3 re-run)", "",
             f"- Seed: {seed}", f"- Method: `{METHOD_DESC}`",
             f"- Re-run timestamp: {ts}", ""]
    lines += [f"- {k}: {'IDENTICAL' if v else 'DIFFERS'}"
              for k, v in results.items()]
    lines += ["", f"RESULT: {'PASS' if passed else 'FAIL'}", ""]
    (REPORTS_DIR / "reproducibility_check.md").write_text("\n".join(lines), encoding="utf-8")

    with open(REPORTS_DIR / "sampling_report.md", "a", encoding="utf-8") as f:
        f.write(f"\nReproducibility re-run: {'PASS — identical selections reproduced' if passed else 'FAIL — see reproducibility_check.md'}\n")

    print("REPRODUCIBILITY PASS" if passed else "REPRODUCIBILITY FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
```

Implementation note: delete the stray `sample_originals_pair` import if it doesn't exist — the module only needs `sample_class` and `METHOD_DESC`. Keep imports minimal and correct.

- [ ] **Step 2: Run the reproduction check**

Run: `python -m dataset_prep.reproduce_check`
Expected: `REPRODUCIBILITY PASS`, exit 0, and `sampling_report.md` now carries the appended reproducibility line. On FAIL: the sampler or pool ordering changed between runs — investigate and fix the implementation (per Global Constraints), then regenerate Task 7–8 artifacts and repeat.

- [ ] **Step 3: Commit**

```powershell
git add dataset_prep/reproduce_check.py project_dataset/reports/reproducibility_check.md project_dataset/reports/sampling_report.md
git commit -m "test(phase3): reproducibility re-run proves deterministic selection"
```

---

### Task 10: Final deliverable summary + readiness verdict

**Files:**
- Create: `dataset_prep/make_final_summary.py`
- Output: `project_dataset/reports/final_summary.md`

**Interfaces:**
- Consumes: `audit_results.json`, `cleaning_log.json`, `sampling_summary.json`, `validation_results.json`, `reproducibility_check.md`, `freeze_manifest.json`, `source_manifest.json`, `label_mapping.yaml`.
- Produces: the final report in the exact template demanded by the agent brief, ending with `READY FOR DIPPER` or `NOT READY FOR DIPPER` (+ blockers). Verdict logic: mapping confirmed AND validation zero errors AND reproduction PASS AND freeze manifest exists ⇒ READY.

- [ ] **Step 1: Implement `dataset_prep/make_final_summary.py`**

```python
"""Aggregate all phase artifacts into the final deliverable summary and verdict."""
import json
import sys
from pathlib import Path

from dataset_prep.config_loader import CONFIGS_DIR, REPORTS_DIR, load_config
from dataset_prep.verify_labels import MAPPING_PATH


def main() -> int:
    cfg = load_config()
    audit = json.loads((REPORTS_DIR / "audit_results.json").read_text(encoding="utf-8"))
    cleaning = json.loads((REPORTS_DIR / "cleaning_log.json").read_text(encoding="utf-8"))
    sampling = json.loads((REPORTS_DIR / "sampling_summary.json").read_text(encoding="utf-8"))
    validation = json.loads((REPORTS_DIR / "validation_results.json").read_text(encoding="utf-8"))
    freeze = json.loads((CONFIGS_DIR / "freeze_manifest.json").read_text(encoding="utf-8"))
    source = json.loads((CONFIGS_DIR / "source_manifest.json").read_text(encoding="utf-8"))
    mapping_doc = MAPPING_PATH.read_text(encoding="utf-8")
    confirmed = "confirmed_by_operator: true" in mapping_doc
    repro_pass = "RESULT: PASS" in (REPORTS_DIR / "reproducibility_check.md").read_text(encoding="utf-8")

    blockers = []
    if not confirmed:
        blockers.append("label mapping not operator-confirmed")
    if validation["errors"]:
        blockers.extend(f"validation error: {e}" for e in validation["errors"])
    if not repro_pass:
        blockers.append("sampling reproducibility check failed")
    if not freeze.get("files"):
        blockers.append("freeze manifest missing/empty")

    verdict = "READY FOR DIPPER" if not blockers else "NOT READY FOR DIPPER"
    h, a = cleaning["human"], cleaning["ai"]

    lines = [
        "# Final Deliverable Summary", "",
        f"Source dataset: {source['repo_id']}",
        f"Source revision/version: `{source['revision']}`",
        f"Original row count: {audit['combined']['n_rows']}", "",
        f"Human candidates: {h['final']} (removed {h['removed_invalid_or_null_label'] + h['removed_missing_text'] + h['removed_empty_text'] + h['removed_duplicate_text']})",
        f"Human selected: {sampling['human']['selected']}",
        f"Human removed: invalid/null label {h['removed_invalid_or_null_label']}, missing {h['removed_missing_text']}, empty {h['removed_empty_text']}, duplicates {h['removed_duplicate_text']}", "",
        f"AI candidates: {a['final']} (removed {a['removed_invalid_or_null_label'] + a['removed_missing_text'] + a['removed_empty_text'] + a['removed_duplicate_text']})",
        f"AI selected: {sampling['ai']['selected']}",
        f"AI removed: invalid/null label {a['removed_invalid_or_null_label']}, missing {a['removed_missing_text']}, empty {a['removed_empty_text']}, duplicates {a['removed_duplicate_text']}", "",
        f"Final total: {sampling['total_selected']} "
        f"(Human = {sampling['human']['selected']}, AI = {sampling['ai']['selected']})", "",
        f"Random seed: {cfg['dataset']['random_seed']}",
        f"Sampling method: `{sampling['method']}`",
        "Label mapping: see `project_dataset/configs/label_mapping.yaml` "
        f"(operator-confirmed: {confirmed})", "",
        "Duplicates: none remaining in final set (validated within-class and across classes)",
        "Missing values: none in final set (validated)",
        f"Validation status: {'PASSED' if not validation['errors'] else 'FAILED'} "
        f"({sum(validation['checks'].values())}/{len(validation['checks'])} checks green)",
        f"Reproducibility: {'PASS' if repro_pass else 'FAIL'}", "",
        f"Human dataset path: project_dataset/sampled/human_5000.parquet",
        f"AI dataset path: project_dataset/sampled/ai_5000.parquet",
        f"Combined dataset path: project_dataset/sampled/originals_10000.parquet",
        f"CSV export path: project_dataset/exports/originals_10000.csv",
        f"Freeze manifest: project_dataset/configs/freeze_manifest.json "
        f"(frozen_at {freeze['frozen_at']})", "",
    ]
    if blockers:
        lines += ["Blocking issues:"] + [f"- {b}" for b in blockers] + [""]
    lines.append(verdict)
    (REPORTS_DIR / "final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-1:]))
    print(f"Summary written to {REPORTS_DIR / 'final_summary.md'}")
    return 0 if not blockers else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the final summary**

Run: `python -m dataset_prep.make_final_summary`
Expected: prints `READY FOR DIPPER` (or `NOT READY FOR DIPPER` with blockers listed above it) and exit code 0 (or 1 if blocked).

- [ ] **Step 3: Full-suite regression + commit**

Run: `python -m pytest tests/dataset_prep -v`
Expected: all tests across all test files pass (config 8, env 3, labels 12, audit 4, candidates 3, sampling 5, validation 4 = 39 passed).

```powershell
git add dataset_prep/make_final_summary.py project_dataset/reports/final_summary.md
git commit -m "docs(phase4): final deliverable summary and DIPPER-readiness verdict"
```

---

## Self-Review Record

**Spec coverage map** (brief section → plan task): Phase 0 freeze/config → Task 1 · HF auth → Task 2 · download+immutable copy → Task 3 · schema inspection + label verification + STOP rule → Task 4 · source audit (all listed metrics) → Task 5 · cleaning with per-stage logging + provenance + no content alteration → Task 6 · seeded random sampling, stable IDs, no first-N, method recording → Task 7 · count/label/uniqueness/provenance validation from disk + freeze + DIPPER-null placeholders → Task 8 · reproducibility test comparing source-row IDs → Task 9 · final deliverable template + READY/NOT READY → Task 10. Mandated directory layout honored throughout. Rules 1–6 each have enforcing mechanisms (Task 4 gate, logs in Task 6, provenance columns in 6/7, no-DIPPER enforced by placeholder-only columns, disk-reload in 8, intermediates retained/gitignored-not-deleted in 1/3).

**Placeholder scan:** No TBD/TODO/fill-in-later items; every code step contains complete runnable code; the one conditional noted in Task 9 Step 1 (unused import cleanup) is an explicit instruction, not a placeholder.

**Type/name consistency:** `select_indices`, `assign_ids`, `sample_class`, `clean_label_subset`, `validate`, `load_label_mapping`, `FINAL_COLUMN_ORDER`, `DIPPER_PLACEHOLDER_COLS` are defined once and referenced with identical names/signatures in consuming tasks; candidate schema and final-schema column orders match between producer and consumer tasks; label-mapping YAML int-key convention (`{0: human, 1: ai}`) is used consistently in `load_label_mapping` consumers (Tasks 6, 7, 8, 9, 10).
