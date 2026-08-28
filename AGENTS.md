# PARAGON Agent Guide

## Project snapshot

PARAGON is a research project on detecting AI-generated English paragraphs after controlled paraphrasing. The intended later system compares existing detectors with graph/GNN and deterministic graph-feature evidence, then evaluates hybrid models by paraphrase intensity.

The implemented work is only dataset preparation (Phases 0-4). A validated corpus of 5,000 human and 5,000 original AI texts has already been generated. DIPPER generation, graph construction, GNNs, detector baselines, and ensemble experiments are planned but are not yet implemented.

Read [README.md](README.md) (mirrored by `PARAGON_MASTER_PROJECT_NEW.md`) before making methodology or scope decisions. The implementation plan in `docs/superpowers/plans/2026-08-24-dataset-prep-phase0-4.md` is aspirational in places; use the checked-in code and artifacts as the record of what currently exists.

## Repository layout

- `project_dataset/run_pipeline.py` - monolithic pipeline that downloads/audits, cleans, samples, validates, and exports the original corpus.
- `project_dataset/configs/sampling_config.yaml` - dataset source, labels, sample counts, seed, quality filters, and output settings.
- `project_dataset/sampled/` - tracked Parquet artifacts: separate human/AI samples and the combined 10,000-row corpus.
- `project_dataset/exports/originals_10000.csv` - CSV form of the combined corpus.
- `project_dataset/reports/` - audit, sampling, validation, final-summary, and pipeline-log evidence for the existing run.
- `dataset_prep/config_loader.py` - small config loader/validator used by tests.
- `tests/dataset_prep/` - pytest coverage for the config-loader contract.

## Dataset invariants

- Source: `andythetechnerd03/AI-human-text`; `text` is the content column and `generated` is the source label.
- Label `0` is human; label `1` is AI. The task remains binary: human `0`, AI `1`.
- Keep exactly 5,000 human and 5,000 AI originals, selected with seed `2026` unless a deliberate, documented dataset revision is requested.
- Preserve `sample_id`, source dataset/split/row provenance, original label, and parent/paraphrase fields. IDs use `H00001`-`H05000` and `A00001`-`A05000`.
- Future paraphrases must remain linked to their original AI sample, and every family must stay in one train/validation/test split to prevent leakage.
- Treat tracked Parquet, CSV, and reports as reproducibility artifacts. Regenerate them only as an intentional dataset revision, then revalidate and update the related reports together.

## Working conventions

- Run tests from the repository root: `python -m pytest -q`.
- The historical config-loader contract accepts both `dataset.source_splits` and the checked-in pipeline's `dataset.splits_to_use`; new configurations should use one schema consistently and tests must cover it.
- `python project_dataset/run_pipeline.py` downloads/uses the Hugging Face dataset and overwrites corpus artifacts and reports. Run it only when intentionally rebuilding the dataset; inspect the resulting validation report before committing changed artifacts.
- Do not read, print, or commit the `HF_TOKEN` in `.env`.
- Keep source text unchanged apart from the documented cleaning rules. Record all removals and any changed sampling, labels, quality thresholds, or data source in reports/configuration.
- Add dependencies to `requirements.txt` when new code requires them, and add focused pytest coverage for new reusable modules.
