# Examples

These example files are designed to be copied, read quickly, and run immediately from the repository root.

## Included demos

### `classification/`

- `config.yaml`: passing binary classification example with probability scores and ROC AUC
- `failing_thresholds.yaml`: intentionally failing version of the same evaluation to demonstrate CI behavior
- `data.csv`: small labeled dataset used by both configs

### `regression/`

- `config.yaml`: passing regression example with `mae`, `rmse`, and `r2`
- `data.csv`: small numeric dataset with realistic prediction error

### `text/`

- `config.yaml`: passing string-output evaluation example
- `data.jsonl`: JSONL records with reference and prediction strings

### `comparison/`

- `classification_baseline.json`: sample baseline result file
- `classification_candidate.json`: sample candidate result file for `orya-eval compare`

## Suggested first commands

```bash
orya-eval run examples/classification/config.yaml
orya-eval run examples/regression/config.yaml
orya-eval run examples/text/config.yaml
orya-eval run examples/classification/failing_thresholds.yaml
orya-eval compare \
  examples/comparison/classification_baseline.json \
  examples/comparison/classification_candidate.json \
  --delta-threshold accuracy=-0.05 \
  --delta-threshold roc_auc=-0.03
```

The failing classification config is expected to exit non-zero. That is intentional and demonstrates the same behavior you would rely on in CI.
