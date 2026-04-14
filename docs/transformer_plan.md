# Transformer Phase Plan

## Goal

Build the first transformer experiment as a strict architectural comparison against the current LSTM latent predictor.

The first transformer should be:

- unimodal
- encoder-only
- fixed-history
- trained on normalized latent embeddings
- evaluated with the same split, metrics, and rollout setup as the current LSTM baseline

The objective is not to build the final best model yet. The objective is to answer one narrow question:

> If we replace the recurrent latent predictor with a transformer, while keeping everything else as similar as possible, do we get better next-step and rollout prediction quality?

## Current Repo State

The current training path already supports:

- precomputed latent embeddings
- fixed or variable history windows
- train/val/holdout fire-level splits
- normalized embedding training
- one-step latent prediction
- rollout evaluation at horizon `H`

The current implemented predictor is an LSTM baseline. The transformer should be introduced as a drop-in replacement for that predictor, not as a new multimodal or end-to-end system.

## First Transformer Scope

### Keep fixed

- input data source
- fire-level split
- history length = `5`
- target = next embedding `z_target`
- normalized embedding setup
- training loss = `MSE`
- rollout evaluation logic
- validation metrics

### Change only

- temporal backbone: `LSTM -> Transformer encoder`

This keeps the comparison clean:

- `LSTM unimodal`
- `Transformer unimodal`

## Model Definition

Treat each latent embedding timestep as one token.

If the batch input is:

```text
(B, T, E)
```

then for the first run:

- `B` = batch size
- `T = 5`
- `E` = embedding dimension, for example `1024` with DINO large

### Minimal architecture

1. Input projection from `E -> d_model`
2. Add positional embeddings
3. Pass tokens through a transformer encoder
4. Read the output at the last timestep
5. Project to the next embedding dimension `E`

Conceptually:

```text
z_t-4, z_t-3, z_t-2, z_t-1, z_t
-> Transformer encoder
-> last token representation
-> linear head
-> z_t+1
```

## Recommended First Config

Use a conservative configuration that is large enough to be meaningful but still close to baseline experimentation.

- `d_model = 512`
- `nhead = 8`
- `num_layers = 4`
- `dim_feedforward = 1024` or `2048`
- `dropout = 0.1`
- output head projects back to `E`

This should be the first implementation unless there is a concrete memory or speed issue.

## Positional Encoding

The model needs positional information because the transformer has no temporal ordering by default.

For the first version, use learned positional embeddings.

Why this is the right first choice here:

- sequence length is short
- implementation is simple
- it matches the goal of a practical baseline comparison

Sinusoidal encoding can be tested later if needed, but it is not necessary for the first run.

## Readout Strategy

Use the output token at the last timestep as the context representation.

This is the cleanest first readout because:

- the task is next-step prediction from past context
- it mirrors the idea that the most recent timestep should aggregate the attended history
- it avoids adding a `CLS` token or extra pooling choices too early

Do not start with:

- mean pooling
- `CLS` token
- more complex summarization heads

Those can become ablations later.

## Training Objective

Keep the loss identical to the baseline:

```text
MSE(pred_z, z_target)
```

Reasons:

- baseline comparison stays fair
- the repo already uses this objective
- the normalized embedding setup makes this a reasonable first objective

Possible later ablations:

- cosine loss
- `MSE + cosine`

But not in the first transformer experiment.

## Rollout Policy

Rollout should match the existing LSTM logic as closely as possible.

At each rollout step:

1. predict the next embedding
2. normalize the predicted embedding if rollout normalization is enabled
3. append prediction to the sliding window
4. continue until the chosen horizon

The rollout normalization behavior should remain aligned with the current cleaned baseline so that the comparison stays about architecture, not evaluation differences.

## Implementation Strategy

### Step 1

Create a new predictor class, for example `TransformerLatentPredictor`, with:

- input projection
- learned positional embeddings
- transformer encoder
- final projection head

### Step 2

Integrate it into the existing training script with minimal branching.

The training script should continue to reuse:

- the same dataset classes
- the same metrics
- the same split logic
- the same rollout code

### Step 3

Expose the transformer through config in the same general style as the current LSTM path.

The first experiment should be easy to run with a config that changes architecture only.

### Step 4

Run a strict comparison against the current normalized LSTM baseline run.

## Experiment Order

### Experiment 1: strict baseline match

Transformer with:

- unimodal latent input only
- history `5`
- normalized embeddings
- same split
- same rollout policy

Only the temporal architecture changes.

This is the most important first experiment.

### Experiment 2: larger temporal window

After the matched comparison is complete, test whether longer context helps:

- history `5`
- history `8`
- history `10`
- optionally full available history if practical

This is where the thesis argument becomes stronger. If longer-context transformer performance improves meaningfully, that supports the claim that wildfire progression is not well explained by only the most recent latent state.

## What To Avoid In This Phase

Do not expand scope yet.

Avoid:

- multimodal transformer inputs
- cross-attention
- patch-level image transformers
- autoregressive decoder-style transformer designs
- decoder integration
- end-to-end spatial prediction

These belong to later phases. Introducing them now would make the baseline comparison weaker and harder to interpret.

## Thesis Framing

This experiment supports a clean scientific question:

> Does a non-recurrent temporal model that can attend across the full latent history improve next-step wildfire latent prediction?

The expected argument structure is:

1. LSTM is the baseline temporal latent predictor.
2. Transformer is introduced as a controlled replacement.
3. If transformer improves one-step and rollout metrics, especially with longer history, that supports the claim that broader temporal context matters.

This is a stronger thesis story than jumping immediately into a large multimodal architecture.

## Suggested Results Table

Start with a compact comparison table like this:

| Model | Input | History | Val Cosine | Rollout Cosine@10 | Val MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| LSTM | embeddings | 5 | ... | ... | ... |
| Transformer | embeddings | 5 | ... | ... | ... |
| Transformer | embeddings | 10 | ... | ... | ... |

Later rows can add multimodal variants, but not yet.

## Bottom Line

The transformer should be developed now as a drop-in replacement for the LSTM latent predictor.

The first phase should stay narrow:

- encoder-only
- unimodal
- fixed history
- normalized embeddings
- same metrics
- same rollout

That is the fastest path to a credible result and a defensible thesis comparison.
