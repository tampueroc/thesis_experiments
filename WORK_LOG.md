# Work Log

## Scope
This log summarizes the substantial engineering work completed in `thesis_experiments` for the wildfire pipeline, focusing on durable design and implementation decisions (not trivial command-by-command activity).

## 1) Core Sequence Dataset Pipeline
- Implemented `WildfireSequenceDataset` with temporal sample construction:
  - `z_in: (history, E)`
  - `z_target: (E,)`
  - `w_in: (history, d_w)`
  - `g: (d_g,)`
  - `fire_id`
- Added validation and indexing behavior (history/stride windowing, shape checks, missing-source checks).
- Added pytest coverage for dataset behavior and sequence windowing logic.

## 2) Real-Data Integration (Embeddings + Weather + Static)
- Added real-data loading pipeline under `src/wildfire/data/real_data.py`.
- Implemented alignment from embedding sequence IDs to weather files and static sources.
- Added real-data smoke test and `RELELA_ONLY_TEST=1` workflow gating for SSH-only tests.
- Validated end-to-end dataset shape behavior on SSH against real embeddings.

## 3) Embedding Precompute Script (Timestamped, Multi-Input)
- Implemented `01_extract_fire_frame_embeddings.py` for organized dataset processing.
- Added support for:
  - `fire_frames` and `isochrones` (single or both)
  - timestamped output runs
  - model-specific output folders
  - full sequence processing (`--all-farmes`)
  - metadata/manifests per run
- Standardized output pattern:
  - `{output}/embeddings/{timestamp}/{input_type}/{model_slug}/{sequence_id}.npy`

## 4) Landscape Analysis Refactor: BBox -> GeoTIFF Channels
- Replaced early bbox-driven analysis with direct `Input_Geotiff.tif` channel analysis.
- Added channel semantics and channel-level summaries.
- Implemented no-data normalization (`-9999 -> -1`) and explicit no-data analytics.
- Split visualization strategy by data type:
  - continuous channels (distribution/range plots)
  - multiclass categorical channels (top-class plots)
  - binary presence/no-data channels (`-1/1`)
- Added timestamped artifact generation for analysis runs.

## 5) Pivoting GeoTIFF to NPY (Operational Data Product)
- Added `05_pivot_landscape_geotiff_to_npy.py`.
- Produces reusable arrays:
  - `landscape_channels_chw.npy`
  - `landscape_channels_hwc.npy`
  - `landscape_pixels_by_channels.npy` (pivot table form)
  - `landscape_nodata_mask_chw.npy`
  - per-channel `.npy` files
  - `meta.json`
- Ensured cleaned no-data representation (`-1` only; no residual `-9999`).

## 6) Sequence-Relative Static g from Pivoted Landscape
- Added `06_build_sequence_static_g_from_landscape.py`.
- Computes per-sequence static `g` vectors using:
  - `indices.json` bounding boxes
  - pivoted landscape tensor (`landscape_channels_chw.npy`)
- Saved outputs in landscape directory:
  - `sequence_static_g_values.npy`
  - `sequence_static_g_keys.json` (alias -> row index)
  - `sequence_static_g_meta.json`
- Integrated this into `build_sources()` with fallback order:
  1. precomputed sequence static (`values + keys`)
  2. legacy `sequence_static_g.npz`
  3. on-the-fly CHW patch summary
  4. legacy bbox geometry vector fallback

## 7) Elevation Reconstruction Artifacts
- Added region-level elevation reconstruction from sequence `g` vectors:
  - `07_render_elevation_heatmap_from_g.py`
- Added pixel-level terrain reconstruction (final preferred visualization):
  - `08_render_elevation_terrain_from_pixels.py`
- Final accepted terrain artifact uses raw pixel elevation channel (`a5`) with value-based coloring and no-data masking.

## 8) SSH/GPU Workflow Hardening
- Established repeatable local->remote delivery workflow:
  1. develop and lightweight-check locally
  2. `uv run ruff check` and `uv run ty check`
  3. commit/push first
  4. SSH fast-forward sync
  5. run heavy jobs/tests on SSH
  6. rsync artifacts back locally
- Used timestamped artifact directories throughout to keep runs auditable and comparable.

## 9) Current Functional Status
- Dataset supports sequence-relative static data from real landscape products.
- Landscape pipeline now has:
  - pivot export
  - static vector generation
  - typed channel analysis plots
  - final pixel-level terrain elevation rendering.
- SSH validation runs completed for new scripts and major integration points.
