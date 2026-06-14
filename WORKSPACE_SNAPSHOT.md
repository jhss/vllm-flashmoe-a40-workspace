# A40 vLLM / FlashMoE Workspace Snapshot

Snapshot date: 2026-06-14

This repository is a source snapshot of the current `/workspace` used for
multi-GPU MoE profiling and optimization work.

Included directories:

- `vllm/`
  - Branch: `codex/flashmoe-a40-moe-profiling`
  - Commit: `cb69b7962ed619446026d3c7c3f15f486af36258`
  - Upstream fork pushed earlier: `https://github.com/jhss/vllm`
- `FlashMoE/`
  - Branch: `codex/cuda13-a40-compat`
  - Commit: `40c51f4429a9563772c3d50039c1d55bd05f4774`
  - Upstream fork pushed earlier: `https://github.com/jhss/FlashMoE`

Excluded from this snapshot:

- Nested `.git/` directories from the source repositories
- Local caches such as `.cache/` and `.flashmoe_cache/`
- Python cache directories
- Build outputs and compiled binaries such as `*.so`, `*.o`, and `*.a`
- Package/build metadata such as `*.egg-info/`, `build/`, and `dist/`

Note: a GitHub token was provided in the chat during setup, but it is not
stored in this snapshot. Rotate that token after this push.
