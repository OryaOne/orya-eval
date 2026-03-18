# orya-eval

**Lightweight evaluation and regression checks for ML and AI systems.**

`orya-eval` is a small, focused Python tool for teams that want repeatable evaluation runs without adopting a heavyweight platform. You define an evaluation in YAML, run it locally or in CI, compare results to a baseline, and fail builds when quality drops.

[![CI](https://img.shields.io/badge/ci-github_actions-black)](./.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](./pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

## 🔎 Overview

ML and AI projects often need something very simple:

- a config file that is easy to review in Git
- a repeatable local command
- structured output for automation
- a clean failure signal when quality regresses

Too many tools jump straight to platforms, dashboards, hosted services, or complex experiment systems. `orya-eval` exists for the narrower, practical job of evaluation and regression checking.

It is designed for engineers who want a tool that is:

- easy to understand on first read
- easy to run in under five minutes
- reliable in CI
- small enough to maintain

## ✨ Features

- `YAML-first workflow`: evaluation definitions stay readable, reviewable, and version-controlled.
- `Focused task support`: built-in evaluation for classification, regression, and string-output systems.
- `CI-ready thresholds`: fail fast when quality drops below a minimum or error rises above a maximum.
- `Clear result artifacts`: write JSON for automation and Markdown for human review.
- `Regression comparison`: compare baseline and candidate runs with explicit metric deltas.
- `Small, typed codebase`: straightforward module boundaries, tests, and GitHub Actions CI from the start.

## 📁 Example Files

All demo commands in this README use committed files from [`examples/`](./examples).

- [`examples/classification/config.yaml`](./examples/classification/config.yaml): passing binary classification example with ROC AUC
- [`examples/classification/failing_thresholds.yaml`](./examples/classification/failing_thresholds.yaml): intentionally failing threshold example for CI behavior
- [`examples/regression/config.yaml`](./examples/regression/config.yaml): passing regression example with `mae`, `rmse`, and `r2`
- [`examples/text/config.yaml`](./examples/text/config.yaml): passing string-output evaluation example
- [`examples/comparison/classification_baseline.json`](./examples/comparison/classification_baseline.json) and [`examples/comparison/classification_candidate.json`](./examples/comparison/classification_candidate.json): sample result files for `compare`

See [`examples/README.md`](./examples/README.md) for a quick explanation of what each demo file is showing.

## ⚡ Quickstart

From the repository root, run these commands exactly:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
orya-eval run examples/classification/config.yaml
```

What happens:

- a compact pass/fail summary is printed to the terminal
- `examples/classification/reports/results.json` is created
- `examples/classification/reports/report.md` is created
- the command exits with code `0`

Example output:

```text
Status: PASS

Summary:
  Command         run
  Task            classification
  Run name        Example classification evaluation
  Rows            6

Metrics:
  accuracy                0.833333
  f1                      0.828571
  precision               0.875000
  recall                  0.833333
  roc_auc                 1.000000

Threshold checks:
  pass accuracy         0.833333 >= 0.800000
  pass roc_auc          1.000000 >= 0.950000
```

To generate a starter config instead of using the included examples:

```bash
orya-eval init --template text --output-dir ./scratch
orya-eval run scratch/orya-eval.text.yaml
```

## 🧪 Examples

This walkthrough uses only the committed demo files in [`examples/`](./examples). A first-time user should be able to copy these commands directly and understand the project quickly.

### 1. Run the classification example

```bash
orya-eval run examples/classification/config.yaml
```

This writes:

- `examples/classification/reports/results.json`
- `examples/classification/reports/report.md`

### 2. Compare a candidate result to a baseline

```bash
orya-eval compare \
  examples/comparison/classification_baseline.json \
  examples/comparison/classification_candidate.json \
  --delta-threshold accuracy=-0.05 \
  --delta-threshold roc_auc=-0.03
```

Example comparison output:

```text
Status: PASS

Summary:
  Command         compare
  Task            classification
  Shared metrics  5
  Regressions     5

Metric deltas:
  accuracy                0.860000 ->   0.830000  delta -0.030000  regression
  f1                      0.860000 ->   0.820000  delta -0.040000  regression
  precision               0.870000 ->   0.840000  delta -0.030000  regression
  recall                  0.860000 ->   0.830000  delta -0.030000  regression
  roc_auc                 0.960000 ->   0.940000  delta -0.020000  regression

Delta thresholds:
  pass accuracy         -0.030000 >= -0.050000
  pass roc_auc          -0.020000 >= -0.030000
```

### 3. See how CI-style threshold failure works

Run the intentionally failing example:

```bash
orya-eval run examples/classification/failing_thresholds.yaml
```

Expected outcome:

- the command prints `Status: FAIL`
- the threshold check section shows which metric failed
- JSON and Markdown artifacts are still written
- the process exits non-zero, which is what CI can use

Example failure output:

```text
Status: FAIL

Summary:
  Command         run
  Task            classification
  Run name        Failing classification threshold demo

Threshold checks:
  fail accuracy         0.833333 >= 0.900000
```

### 4. Compare with stricter regression handling

If you want any regression to fail immediately, use:

```bash
orya-eval compare \
  examples/comparison/classification_baseline.json \
  examples/comparison/classification_candidate.json \
  --fail-on-regression
```

That sample command exits non-zero because every shared metric regresses.

## 🧪 What It Evaluates

`orya-eval` currently supports three task types.

### Classification

Input:

- tabular data from `.csv` or `.jsonl`
- target labels
- predicted labels
- optional probability column for ROC AUC

Metrics:

- `accuracy`
- `precision`
- `recall`
- `f1`
- `roc_auc`

### Regression

Input:

- tabular data from `.csv` or `.jsonl`
- numeric target column
- numeric prediction column

Metrics:

- `mae`
- `rmse`
- `r2`

### Text

Input:

- reference string column
- prediction string column

Metrics:

- `exact_match`
- `contains_match`
- `token_f1`
- `normalized_similarity`

## 🚫 What It Does Not Do

`orya-eval` is intentionally narrow in V1.

It does not include:

- web dashboards
- authentication
- databases
- experiment tracking backends
- MLflow integration
- LLM judge integrations
- hosted services or cloud deployment features
- prompt-testing platform complexity

If you need a reliable local evaluator and CI gate, that is the job it is built to do.

## 📏 Thresholds and CI Behavior

Thresholds are defined in the YAML config for `orya-eval run`.

Example:

```yaml
thresholds:
  accuracy: 0.85
  roc_auc: 0.90
```

Behavior:

- higher-is-better metrics like `accuracy`, `f1`, `r2`, and `exact_match` are treated as minimums
- lower-is-better metrics like `mae` and `rmse` are treated as maximums
- if a threshold fails, `orya-eval run` exits with a non-zero status code

This makes CI usage straightforward:

```bash
orya-eval run examples/classification/failing_thresholds.yaml
```

That example is intentionally configured to fail. It demonstrates exactly how a CI job can stop when model quality drops below an agreed threshold.

For `orya-eval compare`, thresholds are delta-based:

- `--delta-threshold accuracy=-0.01` means accuracy may drop by at most `0.01`
- `--delta-threshold mae=0.05` means MAE may increase by at most `0.05`

You can also skip explicit comparison thresholds and use `--fail-on-regression` to fail on any worse shared metric.

## 🧭 Command Reference

### `orya-eval init`

Create a starter config and small sample dataset.

```bash
orya-eval init --template <classification|regression|text> [--output-dir PATH] [--force]
```

Notes:

- writes a YAML config with report paths already set
- writes matching sample data
- refuses to overwrite existing files unless `--force` is provided

### `orya-eval run`

Run one evaluation from a YAML config.

```bash
orya-eval run <config.yaml>
```

What it does:

1. Loads and validates the config.
2. Loads data from `.csv` or `.jsonl`.
3. Computes the configured metrics.
4. Applies thresholds if present.
5. Writes JSON results.
6. Writes a Markdown report when configured.
7. Exits non-zero if threshold checks fail.

### `orya-eval compare`

Compare two result JSON files.

```bash
orya-eval compare <baseline_results.json> <candidate_results.json> \
  [--delta-threshold metric=value] \
  [--markdown-report PATH] \
  [--fail-on-regression]
```

What it does:

- compares shared metrics across two result files
- prints baseline, candidate, and delta values
- marks regressions clearly
- optionally applies delta thresholds
- optionally writes a Markdown comparison report

## 📝 Config Format

Example classification config:

```yaml
run_name: Example classifier check
description: Smoke test for a binary classification model.
task_type: classification
data_path: data.csv
columns:
  target: target
  prediction: prediction
  probability: probability
metrics:
  - accuracy
  - precision
  - recall
  - f1
  - roc_auc
thresholds:
  accuracy: 0.85
  roc_auc: 0.90
reports:
  json: reports/results.json
  markdown: reports/report.md
metadata:
  model: baseline-v1
```

Core config fields:

- `task_type`: `classification`, `regression`, or `text`
- `data_path`: input dataset path
- `columns`: task-specific column mapping
- `metrics`: optional explicit metric list
- `thresholds`: optional pass/fail criteria
- `reports`: JSON output path and optional Markdown output path
- `run_name` and `description`: human-readable context
- `metadata`: extra string metadata to carry into result files

Included example configs:

- [examples/classification/config.yaml](./examples/classification/config.yaml)
- [examples/regression/config.yaml](./examples/regression/config.yaml)
- [examples/text/config.yaml](./examples/text/config.yaml)

## 🗂 Project Structure

```text
orya_eval/
  cli.py
  comparison.py
  config.py
  runner.py
  exceptions.py
  io/
  metrics/
  models/
  reporting/
  tasks/
examples/
  classification/
  regression/
  text/
  comparison/
tests/
.github/
```

Design notes:

- `orya_eval/cli.py` keeps the user-facing interface thin and readable
- `orya_eval/config.py` owns config parsing and validation
- `orya_eval/tasks/` contains task-specific evaluation logic
- `orya_eval/runner.py` orchestrates the evaluation flow
- `orya_eval/comparison.py` handles baseline vs candidate analysis
- `orya_eval/reporting/` renders Markdown reports

## 🛠 Local Development

Set up the project locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run quality checks:

```bash
make check
```

The current test suite covers the main CLI flows:

- `init` starter generation
- `run` for classification, regression, and text examples
- failing threshold behavior
- `compare` with threshold-based and fail-on-regression behavior

## 🛣 Roadmap

Near-term priorities:

- richer comparison workflows and reporting polish
- more robust config ergonomics and validation messages
- additional built-in example datasets
- broader test coverage around edge cases and error paths

Out of scope for V1:

- external judge models
- cloud services
- dashboards
- experiment tracking systems

## 🤝 Contributing

Contributions are welcome and the project is intentionally easy to work with.

If you want to help:

- open an issue for bugs or focused feature ideas
- keep the project aligned with its narrow scope
- add tests with behavior changes
- update examples or docs when CLI behavior changes

Start with [CONTRIBUTING.md](./CONTRIBUTING.md).

## 📄 License

MIT. See [LICENSE](./LICENSE).
