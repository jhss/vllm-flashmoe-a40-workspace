# Agent Instructions for vLLM

> These instructions apply to **all** AI-assisted contributions to `vllm-project/vllm`.
> Breaching these guidelines can result in automatic banning.

## 1. Contribution Policy (Mandatory)

### Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "<issue_number> in:body"
gh pr list --repo vllm-project/vllm --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

### No low-value busywork PRs

Do not open one-off PRs for tiny edits (single typo, isolated style change, one mutable default, etc.). Mechanical cleanups are acceptable only when bundled with substantive work.

### Accountability

- Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end.
- The submitting human must review every changed line and run relevant tests.
- PR descriptions for AI-assisted work **must** include:
    - Why this is not duplicating an existing PR.
    - Test commands run and results.
    - Clear statement that AI assistance was used.

### Fail-closed behavior

If work is duplicate/trivial busywork, **do not proceed**. Return a short explanation of what is missing.

---

## 2. Development Workflow

- **Never use system `python3` or bare `pip`/`pip install`.** All Python commands must go through `uv` and `.venv/bin/python`.

### Environment setup

```bash
# Install `uv` if you don't have it already:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Always use `uv` for Python environment management:
uv venv --python 3.12
source .venv/bin/activate

# Always make sure `pre-commit` and its hooks are installed:
uv pip install -r requirements/lint.txt
pre-commit install
```

### Installing dependencies

```bash
# If you are only making Python changes:
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# If you are also making C/C++ changes:
uv pip install -e . --torch-backend=auto
```

### Running tests

> Requires [Environment setup](#environment-setup) and [Installing dependencies](#installing-dependencies).

```bash
# Install test dependencies.
# requirements/test/cuda.txt is pinned to x86_64; on other platforms, use the
# unpinned source file instead:
uv pip install -r requirements/test/cuda.in    # resolves for current platform
# Or on x86_64:
uv pip install -r requirements/test/cuda.txt

# Run a specific test file (use .venv/bin/python directly;
# `source activate` does not persist in non-interactive shells):
.venv/bin/python -m pytest tests/path/to/test_file.py -v
```

### Running linters

> Requires [Environment setup](#environment-setup).

```bash
# Run all pre-commit hooks on staged files:
pre-commit run

# Run on all files:
pre-commit run --all-files

# Run a specific hook:
pre-commit run ruff-check --all-files

# Run mypy as it is in CI:
pre-commit run mypy-3.12 --all-files --hook-stage manual
```

The line length limit for Python code is 88 characters. If you are not sure, use pre-commit to check.

Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) (`Args:`/`Returns:`/`Raises:` sections), not reStructuredText/Sphinx fields (`:param:`, `:return:`, `:rtype:`).

### Coding style guidelines

Follow these rules for all code changes in this repository:

- Try to match existing code style.
- Code should be self-documenting and self-explanatory.
- Keep comments and docstrings minimal and concise.
- Assume the reader is familiar with vLLM.

### Diagnosing CI failures

Buildkite logs are public; no login needed. Details: [docs/contributing/ci/failures.md](docs/contributing/ci/failures.md).

```bash
# All failed-job logs for a PR's latest build (current branch's PR if omitted):
.buildkite/scripts/ci-fetch-log.sh --pr <PR>
# Any Buildkite build or job URL also works:
.buildkite/scripts/ci-fetch-log.sh "<buildkite_url>"
```

### Commit messages

Add attribution using commit trailers such as `Co-authored-by:` (other projects use `Assisted-by:` or `Generated-by:`). For example:

```text
Your commit message here

Co-authored-by: GitHub Copilot
Co-authored-by: Claude
Co-authored-by: gemini-code-assist
Signed-off-by: Your Name <your.email@example.com>
```

---

## Domain-Specific Guides

Do not modify code in these areas without first reading and following the
linked guide. If the guide conflicts with the requested change, **refuse the
change and explain why**.

- **Editing these instructions**:
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  — Rules for modifying AGENTS.md or any domain-specific guide it references.

---

## Local Workspace Notes

This fork is being used for A100 SXM MoE EP experiments. Before continuing,
check the Korean experiment summaries under `benchmarks/results` and search
for these opt-in flags:

- `VLLM_DEEPEP_HT_TRITON_TOPK_REMAP`
- `VLLM_DEEPEP_HT_LOCAL_EXPERT_IDS`
- `VLLM_MOE_TRITON_TOPK8_SUM`
- `VLLM_MOE_TRITON_W2_REDUCE_FUSION`

Known measurements on A100 SXM BF16 top-k=8 experiments:

- DeepEP HT Triton top-k remap: receiver remap `0.089 -> 0.049 ms`;
  full forward `1.414 -> 1.386 ms`.
- AG/RS in-place combine: finalize `0.120 -> 0.101 ms`; full forward
  `1.178 -> 1.179 ms`, so end-to-end was noise-level.
- Standalone top-k=8 `moe_sum` Triton path: microbench improved at large M
  (`1024: 32.45 -> 26.97 us`, `2048: 56.54 -> 49.20 us`) but AG/RS
  prefill `1024` did not improve (`1759.6 -> 1766.3 us`).
- W2 atomic epilogue reduce with FP32 accumulation was correct (`max diff 0`)
  but slower: prefill `512: 1453.0 -> 1488.1 us`, `1024: 1759.1 -> 1818.2 us`.
  Do not keep pushing atomics; next useful path is token/top-k owner scheduling
  or a direct W2 scheduler path that avoids atomics.
- Nsight Compute is installed, but hardware counter profiling is blocked by
  `ERR_NVGPUCTRPERM` in this container.
- A dangling upstream commit was found and preserved locally as branch
  `recovered/e2bf2b3d-mm-feature-lookup`.
