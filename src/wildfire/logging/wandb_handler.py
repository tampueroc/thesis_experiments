from __future__ import annotations

from pathlib import Path
from typing import Any


class WandbHandler:
    def __init__(
        self,
        *,
        enabled: bool,
        project: str,
        run_name: str,
        output_dir: Path,
        config: dict[str, Any],
        entity: str = "",
        tags: list[str] | None = None,
        mode: str = "online",
    ) -> None:
        self._enabled = enabled
        self._run: Any | None = None
        self._wandb: Any | None = None

        if not enabled:
            return

        try:
            import wandb  # type: ignore[import-not-found]
        except Exception as exc:
            print(f"[warn] wandb disabled: import failed ({exc})")
            self._enabled = False
            return

        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            entity=entity or None,
            name=run_name,
            dir=str(output_dir),
            config=config,
            tags=tags or None,
            mode=mode,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled and self._run is not None

    def watch_model(self, model: Any, log: str = "gradients", log_freq: int = 100) -> None:
        if not self.enabled or self._wandb is None:
            return
        self._wandb.watch(model, log=log, log_freq=log_freq)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self.enabled or self._run is None:
            return
        self._run.log(metrics, step=step)

    def log_summary(self, payload: dict[str, Any]) -> None:
        if not self.enabled or self._run is None:
            return
        for key, value in payload.items():
            self._run.summary[key] = value

    def finish(self) -> None:
        if not self.enabled or self._run is None:
            return
        self._run.finish()
