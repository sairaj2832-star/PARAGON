# PARAGON Virtual Lab Runbook

Version: `v0.1.0` | Scope: reproducible preparation and staged DIPPER paraphrase inference

## Project and implementation status

PARAGON investigates whether graph-derived structural evidence can complement existing AI-text detectors after controlled paraphrasing. The currently implemented research artifact is the original balanced corpus. The lab harness added in this version validates that corpus and provides safe, staged infrastructure for the next DIPPER stage.

| Area | Status | Evidence |
|---|---|---|
| Dataset preparation | Complete | `project_dataset/reports/FINAL_SUMMARY.txt` records 5,000 human + 5,000 AI originals |
| 10K lab master integration | Implemented; validate/merge before use | `scripts/02_validate_dataset.py`, `scripts/03_merge_dataset.py` |
| Environment/GPU validation | Implemented and run; available 6 GB GPU is below the supported DIPPER profile | `scripts/01_check_environment.py` |
| DIPPER checkpoint access/loading | Implemented; blocked locally by unavailable Hugging Face network/cache | `scripts/04_test_model.py` |
| Smoke, pilot, and full inference | Implemented; not run automatically | `scripts/05_smoke_test.py` through `07_full_inference.py` |
| Graph/GNN, detector baselines, hybrid evaluation | Not implemented | Deliberately outside this lab-harness change |

## Repository and data contract

The original dataset remains in `project_dataset/` and is never overwritten by lab commands.

- Inputs: `project_dataset/sampled/human_5000.parquet` and `project_dataset/sampled/ai_5000.parquet`.
- Lab master: `data/processed/master_10k.csv` (created by the merge script).
- Reports: `data/reports/`.
- Checkpoints: `models/` or a Hugging Face model identifier configured in `configs/inference.yaml`.
- Generated paraphrases: `outputs/inference/`; failures: `outputs/failures/`; logs/reports: `outputs/reports/` and `logs/`.

Every master record preserves `sample_id`, source provenance, text, label, source type, and placeholder lineage fields. The merge adds compatibility aliases: `source_id`, `source_type`, `parent_sample_id`, and `paraphraser`. Null lineage values mean the field is not applicable for the original corpus; they are never invented. Future paraphrases must use the original AI `sample_id` as `parent_sample_id`, retain their source relationship, and stay in the same split family as their parent.

## Requirements and virtual-lab sizing

Use Python 3.10+; the existing project was prepared with Python 3.11. Install base dependencies, then install the CUDA-specific PyTorch build recommended by the lab image before installing `requirements-gpu.txt`.

| Profile | GPU/VRAM | CPU/RAM | Disk | Use |
|---|---:|---:|---:|---|
| Recommended | NVIDIA GPU, 40 GB VRAM | 8 cores, 32 GB RAM | 75 GB free | FP16 checkpoint loading and dependable pilot/full runs |
| Supported fallback | NVIDIA GPU, 16-24 GB VRAM | 4 cores, 16 GB RAM | 50 GB free | INT8/INT4 with batch size 1; validate a pilot first |
| Not recommended | CPU-only or <16 GB VRAM | Any | Any | 11B inference is likely impractical; environment/dataset checks still work |

An 11B-parameter checkpoint needs roughly 22 GB for FP16 weights before runtime overhead, so a 24 GB GPU is a practical lower bound rather than a guarantee. INT8 reduces weight memory to roughly half and INT4 to roughly one quarter, but allocation overhead, activations, sequence length, and the exact virtual-lab driver still matter. Quantization is an experiment setting, not an invisible rescue mechanism: set `model.quantization` to `int8` or `int4` in `configs/inference.yaml` and retain the generated report.

CUDA availability, GPU name/VRAM, package presence, RAM, and free disk space are reported by `01_check_environment.py`. It reports a CPU-only environment as not ready for DIPPER; it does not install drivers or modify the VM.

## Fresh-lab setup

Run these from the repository root.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
# Install the PyTorch CUDA wheel that matches the lab image; see pytorch.org.
pip install -r requirements-gpu.txt
python -m pytest -q
python scripts/01_check_environment.py
```

If the model is gated/private, authenticate using the supported Hugging Face mechanism for the lab account. Do not place or print a token in source code, configs, logs, generated reports, or Git. The existing `.env` is secret material and remains outside the lab package.

## Configuration

- `configs/environment.yaml`: minimum Python, disk/RAM thresholds, and required package checks.
- `configs/dataset.yaml`: frozen source paths, expected class counts, master/report paths, seed, and group-split ratios.
- `configs/inference.yaml`: model identifier or local checkpoint path, dtype, quantization, device map, generation controls, input/output locations, resume policy, and DIPPER control values.

Use `float16` only where the GPU supports it. `bfloat16` may be selected when the GPU supports it. `none`, `int8`, and `int4` are the supported quantization choices. The model loader uses the Transformers `AutoTokenizer`/`AutoModelForSeq2SeqLM` workflow with `device_map`, `torch_dtype`, and `BitsAndBytesConfig` where configured; see the [official Transformers documentation](https://huggingface.co/docs/transformers/index).

Do not modify precision, batch size, diversity controls, or model identifier mid-experiment. Create a new output/checkpoint/report path (or archive the prior run) for a different configuration.

## Dataset procedure

```powershell
python scripts/02_validate_dataset.py
python scripts/03_merge_dataset.py
python scripts/02_validate_dataset.py
```

The source validator requires each input class to contain 5,000 records with non-empty text, unique IDs, labels only in `{0, 1}`, and no missing required columns. The merge requires exactly 10,000 records with 5,000 labels `0` (human) and 5,000 labels `1` (AI). It writes:

- `data/processed/master_10k.csv`
- `data/reports/dataset_report.json`
- `data/reports/dataset_report.txt`

The report additionally displays duplicate text counts. Duplicates do not replace the distinct-ID validation; investigate any newly introduced cross-source duplication before experimental use.

To create future leakage-safe splits, call `paragon_lab.dataset.create_splits.create_grouped_splits` with the merged frame and `configs/dataset.yaml` settings. Its grouping key is `parent_sample_id`, falling back to `sample_id`; this keeps an AI original and every generated descendant in one split.

## Model check and staged inference

Never begin with full inference.

```powershell
# Resolves model config and tokenizer/checkpoint access; does not load all weights.
python scripts/04_test_model.py

# Loads the full checkpoint only after the lightweight check passes.
python scripts/04_test_model.py --load

# Run exactly one eligible AI-original record.
python scripts/05_smoke_test.py

# Escalate deliberately: 5, 10, 50, then 100 samples.
python scripts/06_pilot_inference.py --samples 5
python scripts/06_pilot_inference.py --samples 10
python scripts/06_pilot_inference.py --samples 50
python scripts/06_pilot_inference.py --samples 100

# Only after reviewing the 100-sample pilot and report:
python scripts/07_full_inference.py --confirm-full-run
```

The smoke test loads the configured model, tokenizes one input, generates output, rejects empty output, records input/output differences and timing, and saves `outputs/pilot/smoke_test.json`. The pilot/full runner records sample ID, input/output token counts, generation time, model configuration, success status, and failures. Its input defaults to `source_type: ai_original`, so a full run targets the 5,000 original AI rows—not all 10,000 rows.

## Resume, errors, and validation

Successful rows append incrementally to `outputs/inference/dipper_outputs.csv`. `outputs/inference/checkpoint.json` records the successful count. On restart, successful `sample_id`s are read from the output and skipped, so a run interrupted after 2,000 successes resumes the remaining work. The runner will not overwrite a prior output when `--restart` is passed; archive/move the previous run before intentionally regenerating outputs.

Every recoverable sample/batch failure is recorded in `outputs/failures/failed_samples.csv` with timestamp, sample ID, error type/message, and full inference configuration. `KeyboardInterrupt` preserves already appended work. Disk writes use an atomic checkpoint update.

CUDA OOM is explicit: memory is cleared, then batch size may be reduced only when `inference.oom.auto_reduce_batch_size` is true. Each reduction is written into the run report. At batch size 1, the sample fails and is logged; the harness never silently switches FP16 to INT8/INT4. To use a quantized fallback, edit the configuration, start a separately named output/checkpoint/report, and rerun the smoke test.

After a pilot or full run, validate output from Python:

```powershell
python -c "from pathlib import Path; from paragon_lab.evaluation.output_validation import validate_output; print(validate_output(Path('outputs/inference/dipper_outputs.csv')))"
```

Run this from an activated environment with the `src` directory importable, or execute it as `python -c "import sys; sys.path.insert(0, 'src'); ..."`. It reports duplicate successful IDs and empty generations. Use `paragon_lab.evaluation.inference_statistics.inference_statistics` for failure rate and average token/time statistics.

## Troubleshooting

| Failure | Meaning | Action |
|---|---|---|
| Missing package / `ModuleNotFoundError` | Base or GPU dependencies are incomplete | Activate `.venv`, install the appropriate requirements, then rerun environment check |
| CUDA unavailable | CPU PyTorch, no GPU assignment, or driver mismatch | Confirm virtual-lab GPU allocation and install the matching PyTorch CUDA wheel |
| Checkpoint/tokenizer access fails | Invalid path, network restriction, missing auth, or cache miss | Run `04_test_model.py`, verify `name_or_path`, access, and local cache; do not start a pilot |
| CUDA OOM | Model/sequence/batch exceeds available VRAM | Preserve the failure report; reduce configured batch size or create an explicitly quantized experiment |
| Master dataset missing | Merge stage was skipped | Run `03_merge_dataset.py`, then validate it |
| Corrupt/empty output | A prior process did not produce a usable row | Run output validation, inspect failure CSV, and resume after correcting the cause |
| Disk full | Incremental output/checkpoint cannot be saved | Free or attach storage before retrying; do not delete the source artifacts |

## Version history

### v0.1.0

- Added isolated `src/paragon_lab` lab package, config files, command wrappers, and lightweight tests.
- Added lineage-preserving merge/validation, environment reporting, checkpoint checks, staged inference, resume checkpoints, failure logging, and output validation.
- Preserved the existing dataset-generation pipeline and tracked corpus artifacts.

## Operational workflow

1. Check environment and GPU: `python scripts/01_check_environment.py`.
2. Validate the 5K + 5K source artifacts: `python scripts/02_validate_dataset.py`.
3. Merge the master 10K and inspect its reports: `python scripts/03_merge_dataset.py`.
4. Revalidate the master: `python scripts/02_validate_dataset.py`.
5. Check model checkpoint/tokenizer: `python scripts/04_test_model.py`.
6. Load-test the model: `python scripts/04_test_model.py --load`.
7. Run the one-sample smoke test: `python scripts/05_smoke_test.py`.
8. Run staged pilots: 5, 10, 50, then 100 samples.
9. Inspect output validation, failures, quality, and estimated run time.
10. Start the resumable full run only with `--confirm-full-run`.
11. Validate outputs and produce final experiment statistics.

## Current completion status

Update these after the command and its expected checks have passed on the target lab.

- [x] Repository and existing artifacts inspected
- [x] Environment checker implemented and locally tested (current machine is not DIPPER-ready)
- [x] Dataset validator and 5K + 5K merge implemented and tested against available artifacts
- [x] Dataset report generated
- [x] GPU/CUDA checker implemented and run against the available GPU (capacity is insufficient for DIPPER)
- [ ] Model checkpoint and loader tested with an accessible DIPPER checkpoint
- [ ] Smoke test passed
- [ ] Pilot inference passed
- [ ] Output validation passed on real model output
- [x] Error logging and resume behavior tested with lightweight automated tests
- [ ] Full inference completed
- [x] Automated tests passed
- [x] Runbook completed and verified against the available repository artifacts
