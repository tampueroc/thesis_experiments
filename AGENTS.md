# AGENTS.md

## Delivery Gate

For every delivery, run these checks from the project root and fix issues before responding:

1. `uv run ruff check`
2. `uv run ty check`

If either command cannot be run due to environment or permission constraints, report that explicitly in the delivery response.

## Execution Policy

- Use `uv` for all local commands (`python`, `pytest`, lint, type checks).
- Use `uv` for all remote commands as well (for example `/home/tampuero/.local/bin/uv run ...` on SSH host).
- Local machine is for code editing and lightweight checks.
- SSH machine is for heavy-duty GPU/data-scale checks.

## Delivery Methodology

1. Implement code changes locally.
2. Run lightweight local validation with `uv`:
   - `uv run ruff check`
   - `uv run ty check`
   - `uv run --with pytest pytest` (or project pytest env if available)
3. Commit locally.
4. Push local changes to origin and wait for successful push confirmation.
5. Only after push succeeds, run SSH git sync/pull.
6. On SSH machine:
   - `git pull`
   - verify commit hash alignment (`git rev-parse HEAD` on both sides).
7. Run end-to-end heavy checks on SSH with real data using `uv`.
8. Delivery is closed only after local + SSH validations pass.

## SSH-Only Real-Data Tests

- Real-data pytest cases are gated by `RELELA_ONLY_TEST=1`.
- Default local runs should keep these tests skipped.
- Run on SSH machine:
  - `RELELA_ONLY_TEST=1 /home/tampuero/.local/bin/uv run --with pytest pytest -q -m real_data`
