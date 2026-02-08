# Wildfire Config Layout

- `configs/wildfire/latent_predictor/<family>/<variant>_<profile>.toml`
- `configs/wildfire/latent_decoder/<family>/<variant>_<profile>.toml`

Each run should set:
- `component`
- `family`
- `variant`

Artifacts are written to:
- `artifacts/wildfire/<component>/<family>/<run_id>/`

W&B project:
- `latent_wildfire`
