# Isochrone Model 2 Ground-Truth Commands

Run from the repository root.

This trains the ground-truth Model 2 decoder on isochrone images and isochrone embeddings using the same binarized setup as `script 18`.

## GT Isochrone Decoder

```bash
PYTHONPATH=src uv run --with wandb python src/wildfire/scripts/model2_train/18_train_latent_decoder_conditional_unet_binary_binarized.py --config configs/wildfire/latent_decoder/conditional_unet_binary/model_01_binarized_05_04_isochrones.toml --embeddings-model-slug facebook__dinov2-large
```

## Contract

The trained decoder uses:

- previous isochrone image at time `t`
- previous isochrone embedding at time `t`
- target isochrone embedding at time `t+1`

to reconstruct:

- target isochrone image at time `t+1`

In compact form:

`i_t^iso, z_t^iso, z_(t+1)^iso -> i_(t+1)^iso`

## Notes

- no new dataset class is required, because `src/wildfire/data/decoder_dataset.py` and `src/wildfire/scripts/model2_train/18_train_latent_decoder_conditional_unet_binary_binarized.py` already support `input_type = "isochrones"`
- this is the ground-truth latent version of Model 2
- after selecting the best isochrone Model 1 run, the predicted-latent pipeline can be evaluated separately
