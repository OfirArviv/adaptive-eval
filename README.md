# Efficient-Evaluation

## Paper

**Stop Guessing When to Stop Testing: Efficient Model Evaluation with Just Enough Data**

Paper PDF: [Stop Guessing When to Stop Testing: Efficient Model Evaluation with Just Enough Data](https://arxiv.org/abs/2607.08522)

## Abstract

Fixed-size benchmarks are inefficient in both directions: they often waste evaluation budget on examples that add little new information, and they can still fail to resolve close model comparisons even after the full benchmark is consumed. This repository studies adaptive evaluation procedures that decide how much data to use, when to stop, and what claims the available evidence actually supports. The core case studies quantify precision growth under increasing budget, benchmark-level resolution for top model pairs, and pairwise separability under sequential testing on the OpenVLM leaderboard.

## Repository Structure

- `main.py`: paper-facing entrypoint for the OpenVLM studies
- `case_studies/ci_width_precision_curve.py`: precision-versus-budget study
- `case_studies/benchmark_resolution.py`: full-benchmark resolution analysis
- `case_studies/pairwise_separability.py`: sequential pairwise stopping study
- `outputs/case_studies/`: saved configs, csv outputs, narratives, and figures

## Setup

See the full installation guide:

- [INSTALL.md](INSTALL.md)

After installing R, install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Run the Studies

Run the full OpenVLM paper study bundle:

```bash
python -m main
```

This runs the three configured studies on `open_vlm_leaderboard`:

- `ci_width_precision_curve`
- `benchmark_resolution`
- `pairwise_separability`

Artifacts are written under `outputs/case_studies/`, one directory per study and run configuration.

## Study Configs

The default paper-facing configs live in [main.py](main.py):

- `CI_WIDTH_PAPER_CONFIG`
- `BENCHMARK_RESOLUTION_PAPER_CONFIG`
- `PAIRWISE_SEPARABILITY_PAPER_CONFIG`

These are the settings used by `python -m main`.

## Run Programmatically

If you want to run the bundle from Python and inspect the `CaseStudyResult` objects directly:

```python
from main import run_open_vlm_paper_studies

results = run_open_vlm_paper_studies(use_cache=True, verbose=True)
for study_name, result in results.items():
    print(study_name, len(result.raw_rows), len(result.summary_rows))
```

To save the results from Python:

```python
from main import save_open_vlm_paper_studies

paths = save_open_vlm_paper_studies()
for study_name, artifact_paths in paths.items():
    print(study_name, artifact_paths.root_dir)
```

## Notes

- The public entrypoint is focused on the OpenVLM paper studies rather than a generic benchmark runner.
- The benchmark data loader path used by the paper entrypoint is `open_vlm_leaderboard`.
- Pairwise and sequential studies can take time; leave `verbose=True` if you want stage and replicate progress output.
