from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np


SEQ_RE = re.compile(r"sequence_(\d+)$")


def choose_timestamp(embeddings_root: Path, requested: str) -> str:
    if requested:
        return requested
    timestamps = sorted(p.name for p in embeddings_root.iterdir() if p.is_dir())
    if not timestamps:
        raise RuntimeError(f"no timestamp folders found in {embeddings_root}")
    return timestamps[-1]


def parse_sequence_num(sequence_id: str) -> int:
    match = SEQ_RE.match(sequence_id)
    if match is None:
        raise ValueError(f"invalid sequence_id format: {sequence_id}")
    return int(match.group(1))


def load_weather_history(landscape_dir: Path) -> list[str]:
    weather_history_path = landscape_dir / "WeatherHistory.csv"
    with weather_history_path.open(encoding="utf-8") as fh:
        lines = [line.strip() for line in fh if line.strip()]
    if not lines:
        raise RuntimeError(f"empty weather history: {weather_history_path}")
    return lines


def weather_file_from_history(landscape_dir: Path, weather_ref: str) -> Path:
    candidate = landscape_dir / "Weathers" / Path(weather_ref).name
    if not candidate.exists():
        raise FileNotFoundError(f"weather file not found for ref={weather_ref}: {candidate}")
    return candidate


def read_weather_rows(weather_file: Path) -> list[tuple[float, float, float]]:
    rows: list[tuple[float, float, float]] = []
    with weather_file.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ws = float(row.get("WS", "0") or 0)
            wd = float(row.get("WD", "0") or 0)
            fire_scenario = float(row.get("FireScenario", "0") or 0)
            rows.append((ws, wd, fire_scenario))
    if not rows:
        rows = [(0.0, 0.0, 0.0)]
    return rows


def expand_weather_to_timesteps(rows: list[tuple[float, float, float]], timesteps: int) -> np.ndarray:
    out = np.zeros((timesteps, 3), dtype=np.float32)
    for t in range(timesteps):
        idx = t if t < len(rows) else len(rows) - 1
        out[t] = rows[idx]
    return out


def static_vector_from_bbox(bbox: list[int] | tuple[int, int, int, int]) -> np.ndarray:
    row_start, row_end, col_start, col_end = [float(x) for x in bbox]
    height = row_end - row_start
    width = col_end - col_start
    center_row = row_start + height / 2.0
    center_col = col_start + width / 2.0
    return np.asarray(
        [row_start, row_end, col_start, col_end, height, width, center_row, center_col],
        dtype=np.float32,
    )


def build_sources(
    model_dir: Path,
    landscape_dir: Path,
    max_sequences: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    sequences = manifest.get("sequences", [])
    if not sequences:
        raise RuntimeError(f"no sequences in manifest: {model_dir / 'manifest.json'}")

    weather_history = load_weather_history(landscape_dir)
    indices = json.loads((landscape_dir / "indices.json").read_text(encoding="utf-8"))

    z_by_fire: dict[str, np.ndarray] = {}
    w_by_fire: dict[str, np.ndarray] = {}
    g_by_fire: dict[str, np.ndarray] = {}

    for seq in sequences[:max_sequences]:
        sequence_id = seq["sequence_id"]
        sequence_num = parse_sequence_num(sequence_id)
        z = np.asarray(np.load(seq["sequence_path"]), dtype=np.float32)
        if z.ndim != 2:
            raise ValueError(f"{sequence_id}: expected 2D embedding, got {z.shape}")

        if not (1 <= sequence_num <= len(weather_history)):
            continue

        weather_ref = weather_history[sequence_num - 1]
        weather_file = weather_file_from_history(landscape_dir, weather_ref)
        weather_rows = read_weather_rows(weather_file)
        w = expand_weather_to_timesteps(weather_rows, timesteps=z.shape[0])

        bbox = indices.get(str(sequence_num), [0, 0, 0, 0])
        g = static_vector_from_bbox(bbox)

        z_by_fire[sequence_id] = z
        w_by_fire[sequence_id] = w
        g_by_fire[sequence_id] = g

    if not z_by_fire:
        raise RuntimeError("no aligned sequences were built for dataset smoke test")

    return z_by_fire, w_by_fire, g_by_fire
