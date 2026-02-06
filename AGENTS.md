# AGENTS.md

## Delivery Gate

For every delivery, run these checks from the project root and fix issues before responding:

1. `uv run ruff check`
2. `uv run ty check`

If either command cannot be run due to environment or permission constraints, report that explicitly in the delivery response.
