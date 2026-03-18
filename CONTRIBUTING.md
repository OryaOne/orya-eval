# Contributing to orya-eval

Thanks for your interest in contributing.

`orya-eval` is intentionally small and focused. The best contributions improve evaluation workflows, regression checking, CLI usability, documentation, or test coverage without expanding the project beyond its scope.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Or use:

```bash
make install
```

## Run checks

Run the full local check suite before opening a pull request:

```bash
make check
```

Or run commands individually:

```bash
ruff format .
ruff check .
pytest
```

## Proposing changes

- Open an issue first for larger changes or scope questions.
- Keep pull requests focused on one logical change.
- Explain the user-visible impact clearly.
- Include tests for fixes and new behavior.
- Update docs and examples when CLI or config behavior changes.

## Project conventions

- Keep public APIs typed and easy to understand.
- Prefer straightforward modules over clever abstractions.
- Keep CLI output concise, consistent, and actionable.
- Keep evaluation behavior deterministic and local-first.
- Avoid external service dependencies in core flows.

## Project layout

- `orya_eval/config.py` handles config parsing and validation.
- `orya_eval/tasks/` contains task-specific evaluation logic.
- `orya_eval/reporting/` contains Markdown reporting helpers.
- `tests/` should stay readable, focused, and easy to extend.

## Need help?

Open a bug report or feature request in GitHub issues, and include a minimal reproduction when possible.
