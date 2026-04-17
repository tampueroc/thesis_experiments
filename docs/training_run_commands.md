# Training Run Commands

Run these commands from the repository root.

## LSTM

### Normalized long run

```bash
PYTHONPATH=src uv run --with wandb python src/wildfire/scripts/model1_train/09_train_latent_lstm.py --config configs/wildfire/latent_predictor/lstm/model_01_long_run_normalized.toml --embeddings-model-slug facebook__dinov2-large
```

## Transformer

### Base

```bash
PYTHONPATH=src uv run --with wandb python src/wildfire/scripts/model1_train/09_train_latent_lstm.py --config configs/wildfire/latent_predictor/transformer/model_01_base.toml --embeddings-model-slug facebook__dinov2-large
```

### Normalized long run

```bash
PYTHONPATH=src uv run --with wandb python src/wildfire/scripts/model1_train/09_train_latent_lstm.py --config configs/wildfire/latent_predictor/transformer/model_01_long_run_normalized.toml --embeddings-model-slug facebook__dinov2-large
```

### Residual normalized long run

```bash
PYTHONPATH=src uv run --with wandb python src/wildfire/scripts/model1_train/13_train_latent_transformer_residual.py --config configs/wildfire/latent_predictor/transformer/model_01_residual_long_run_normalized.toml --embeddings-model-slug facebook__dinov2-large
```

### Static normalized long run

```bash
PYTHONPATH=src uv run --with wandb python src/wildfire/scripts/model1_train/10_train_latent_transformer_static.py --config configs/wildfire/latent_predictor/transformer/model_01_static_long_run_normalized.toml --embeddings-model-slug facebook__dinov2-large
```

### Static head normalized long run

```bash
PYTHONPATH=src uv run --with wandb python src/wildfire/scripts/model1_train/11_train_latent_transformer_static_head.py --config configs/wildfire/latent_predictor/transformer/model_01_static_head_long_run_normalized.toml --embeddings-model-slug facebook__dinov2-large
```

### Static FiLM head normalized long run

```bash
PYTHONPATH=src uv run --with wandb python src/wildfire/scripts/model1_train/12_train_latent_transformer_static_film.py --config configs/wildfire/latent_predictor/transformer/model_01_static_film_head_long_run_normalized.toml --embeddings-model-slug facebook__dinov2-large
```

### Static FiLM head residual normalized long run

```bash
PYTHONPATH=src uv run --with wandb python src/wildfire/scripts/model1_train/14_train_latent_transformer_static_film_residual.py --config configs/wildfire/latent_predictor/transformer/model_01_static_film_head_residual_long_run_normalized.toml --embeddings-model-slug facebook__dinov2-large
```

## Isochrones

### LSTM normalized long run

```bash
PYTHONPATH=src uv run --with wandb python src/wildfire/scripts/model1_train/09_train_latent_lstm.py --config configs/wildfire/latent_predictor/lstm/model_01_long_run_normalized_isochrones.toml --embeddings-model-slug facebook__dinov2-large
```

### Transformer normalized long run

```bash
PYTHONPATH=src uv run --with wandb python src/wildfire/scripts/model1_train/09_train_latent_lstm.py --config configs/wildfire/latent_predictor/transformer/model_01_long_run_normalized_isochrones.toml --embeddings-model-slug facebook__dinov2-large
```

### Transformer residual normalized long run

```bash
PYTHONPATH=src uv run --with wandb python src/wildfire/scripts/model1_train/13_train_latent_transformer_residual.py --config configs/wildfire/latent_predictor/transformer/model_01_residual_long_run_normalized_isochrones.toml --embeddings-model-slug facebook__dinov2-large
```

### Transformer static normalized long run

```bash
PYTHONPATH=src uv run --with wandb python src/wildfire/scripts/model1_train/10_train_latent_transformer_static.py --config configs/wildfire/latent_predictor/transformer/model_01_static_long_run_normalized_isochrones.toml --embeddings-model-slug facebook__dinov2-large
```

### Transformer static head normalized long run

```bash
PYTHONPATH=src uv run --with wandb python src/wildfire/scripts/model1_train/11_train_latent_transformer_static_head.py --config configs/wildfire/latent_predictor/transformer/model_01_static_head_long_run_normalized_isochrones.toml --embeddings-model-slug facebook__dinov2-large
```

### Transformer static FiLM head normalized long run

```bash
PYTHONPATH=src uv run --with wandb python src/wildfire/scripts/model1_train/12_train_latent_transformer_static_film.py --config configs/wildfire/latent_predictor/transformer/model_01_static_film_head_long_run_normalized_isochrones.toml --embeddings-model-slug facebook__dinov2-large
```

### Transformer static FiLM head residual normalized long run

```bash
PYTHONPATH=src uv run --with wandb python src/wildfire/scripts/model1_train/14_train_latent_transformer_static_film_residual.py --config configs/wildfire/latent_predictor/transformer/model_01_static_film_head_residual_long_run_normalized_isochrones.toml --embeddings-model-slug facebook__dinov2-large
```
