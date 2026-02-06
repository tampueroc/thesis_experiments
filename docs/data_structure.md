# Data Structure

## Raw Input Data (Remote)

Base path:

`/home/tampuero/data/deep_crown_dataset/organized_spreads/`

Expected structure:

```text
organized_spreads/
  fire_frames/
    sequence_0001/
      fire_000.png
      fire_001.png
      ...
  isochrones/
    sequence_0001/
      iso_000.png
      iso_001.png
      ...
```

Notes:
- `fire_frames/` and `isochrones/` are parallel sequence trees.
- Sequence folder names should match across both modalities.
- We now process with `--all-farmes` to embed all available frame indices per sequence.

## Thesis Data (Remote)

Base path:

`/home/tampuero/data/thesis_data/`

Current structure:

```text
thesis_data/
  embeddings/
    <timestamp_iso>/
      meta.json
      fire_frames/
        <model_slug>/
          manifest.json
          sequence_0001.npy
          sequence_0002.npy
          ...
      isochrones/
        <model_slug>/
          manifest.json
          sequence_0001.npy
          sequence_0002.npy
          ...
  landscape/
    Input_Geotiff.tif
    WeatherHistory.csv
    Weathers/
    indices.json
```

Where:
- `<timestamp_iso>` example: `2026-02-06T11-23-27Z`
- `<model_slug>` examples:
  - `facebook__dinov2-small`
  - `facebook__dinov2-large`

## Embeddings Metadata

Each timestamp folder includes:

- `meta.json`: run-level metadata, including model entries and embedding dimensions.

Each modality/model folder includes:

- `manifest.json`: sequence index for that modality/model.
- `sequence_<id>.npy`: per-sequence embedding array.

Per-sequence embedding shape:

- `(T, E)` where:
  - `T` = number of embedded frames for that sequence
  - `E` = encoder embedding dimension (for example, `384` for DINOv2-small, `1024` for DINOv2-large)

## Landscape Data Formats

Remote path:

`/home/tampuero/data/thesis_data/landscape/`

Observed files and formats:

- `Input_Geotiff.tif`
  - TIFF image
  - width: `1406`
  - height: `1173`
- `WeatherHistory.csv`
  - One path per row (string values), for example:
    - `data/risk_study/Weathers/Weather1.csv`
    - `data/risk_study/Weathers/Weather14.csv`
- `indices.json`
  - JSON object mapping string fire ids to 4-int bounding boxes:
  - schema: `{ "<fire_id>": [row_start, row_end, col_start, col_end], ... }`
  - example: `"1": [380, 780, 0, 400]`
