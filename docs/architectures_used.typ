= Architectures Used For Latent Prediction

This document summarizes the latent temporal predictors used in this repository for wildfire spread forecasting in embedding space.

== Shared Setup

At each time step $t$, the fire mask is encoded into a latent vector

$z_t in RR^E$

and the model receives a history window

$z_(t-h+1:t)$

with $h = 5$ in the fixed-window runs.

Some architectures also consume static landscape covariates:

- numeric static features: $g_(text("num"))$
- numeric missingness mask: $m_(text("num"))$
- categorical static features: $g_(text("cat"))$

The prediction target is either the next latent state

$hat(z)_(t+1)$

or a residual update

$hat(Delta z)_(t+1)$

with

$hat(z)_(t+1) = z_t + hat(Delta z)_(t+1)$

== 1. LSTM Predictor

File:

- `src/wildfire/model_latent_predictor/model_01.py`

This is the baseline recurrent model. A sequence of latent vectors is passed through an LSTM, and the final hidden state is projected to the prediction space.

$h_(1:h) = text("LSTM")(z_(t-h+1:t))$

$hat(z)_(t+1) = W h_h + b$

Key characteristics:

- temporal backbone: multi-layer LSTM
- prediction head: linear projection from final hidden state
- used for the `lstm/model_01*` configs
- training loss is direct latent MSE on $hat(z)_(t+1)$

== 2. Plain Transformer Predictor

File:

- `src/wildfire/model_latent_predictor/transformer_01.py`

This model projects each latent input token into a transformer width $d_(text("model"))$, adds learned positional embeddings, encodes the sequence with a transformer encoder, and projects the final token representation to the output embedding.

$x_i = W_(text("in")) z_i$

$tilde(x)_i = x_i + p_i$

$H = text("TransformerEncoder")(tilde(x)_(t-h+1:t))$

$hat(z)_(t+1) = W_(text("out")) H_h$

Key characteristics:

- temporal backbone: transformer encoder
- learned positional encoding over the history window
- no static conditioning
- used for `transformer/model_01_*` configs with `family = "transformer"`

== 3. Residual Transformer Predictor

Trainer:

- `src/wildfire/scripts/model1_train/13_train_latent_transformer_residual.py`

Model backbone:

- `src/wildfire/model_latent_predictor/transformer_01.py`

The residual transformer uses the same plain transformer backbone, but it is trained to predict the latent change instead of the next latent directly.

$hat(Delta z)_(t+1) = f_(theta)(z_(t-h+1:t))$

$hat(z)_(t+1) = z_t + hat(Delta z)_(t+1)$

Key characteristics:

- same encoder architecture as the plain transformer
- different prediction target and loss interpretation
- useful when temporal evolution is easier to model as an increment
- used for `transformer/model_01_residual_*`

== 4. Static-Conditioned Transformer

File:

- `src/wildfire/model_latent_predictor/transformer_static_01.py`

This architecture augments the transformer with static landscape information. Static numeric inputs, their missingness mask, and categorical embeddings are encoded into a static context vector and added to every transformer token before sequence encoding.

$s = phi([g_(text("num")) | m_(text("num")) | text("embed")(g_(text("cat")))])$

$H = text("TransformerEncoder")(tilde(x)_(t-h+1:t) + s)$

$hat(z)_(t+1) = W_(text("out")) H_h$

Key characteristics:

- static features are injected before the encoder
- conditioning acts as a global context shift for all time steps
- used for `transformer/model_01_static_*`

== 5. Static-Head Transformer

File:

- `src/wildfire/model_latent_predictor/transformer_static_head_01.py`

This model keeps the temporal transformer stream and the static stream separate until the prediction head. The transformer encodes only the temporal latent history, while static features are encoded in parallel and concatenated with the final temporal state.

$h = text("TransformerEncoder")(tilde(x)_(t-h+1:t))_h$

$s = phi([g_(text("num")) | m_(text("num")) | text("embed")(g_(text("cat")))])$

$hat(z)_(t+1) = psi([h | s])$

Key characteristics:

- late fusion between temporal and static information
- cleaner separation between temporal dynamics and static context
- used for `transformer/model_01_static_head_*`

== 6. Static FiLM-Head Transformer

File:

- `src/wildfire/model_latent_predictor/transformer_static_film_head_01.py`

This model uses Feature-wise Linear Modulation (FiLM) to condition the final temporal representation with static features. Static covariates generate scale and shift parameters that modulate the final transformer state before output projection.

$h = text("TransformerEncoder")(tilde(x)_(t-h+1:t))_h$

$(gamma, beta) = phi([g_(text("num")) | m_(text("num")) | text("embed")(g_(text("cat")))])$

$h' = (1 + gamma) h + beta$

$hat(z)_(t+1) = W_(text("out")) h'$

Key characteristics:

- static features control the final temporal representation multiplicatively and additively
- more expressive than simple concatenation
- used for `transformer/model_01_static_film_head_*`

== 7. Static FiLM-Head Residual Transformer

Trainer:

- `src/wildfire/scripts/model1_train/14_train_latent_transformer_static_film_residual.py`

Model backbone:

- `src/wildfire/model_latent_predictor/transformer_static_film_head_01.py`

This is the residual version of the static FiLM-head transformer. The architecture is the same as the FiLM-head model, but the predicted quantity is the latent delta.

$hat(Delta z)_(t+1) = f_(theta)(z_(t-h+1:t), g_(text("num")), m_(text("num")), g_(text("cat")))$

$hat(z)_(t+1) = z_t + hat(Delta z)_(t+1)$

Key characteristics:

- combines static FiLM conditioning with residual prediction
- used for `transformer/model_01_static_film_head_residual_*`

== 8. Architecture Summary

#table(
  columns: 4,
  [Variant], [Temporal Backbone], [Static Conditioning], [Prediction Form],
  [LSTM], [LSTM], [None], [Direct $hat(z)_(t+1)$],
  [Transformer], [Transformer encoder], [None], [Direct $hat(z)_(t+1)$],
  [Residual Transformer], [Transformer encoder], [None], [Residual $z_t + hat(Delta z)$],
  [Static Transformer], [Transformer encoder], [Add static context to tokens], [Direct $hat(z)_(t+1)$],
  [Static Head Transformer], [Transformer encoder], [Late concat at head], [Direct $hat(z)_(t+1)$],
  [Static FiLM-Head Transformer], [Transformer encoder], [FiLM on final state], [Direct $hat(z)_(t+1)$],
  [Static FiLM-Head Residual], [Transformer encoder], [FiLM on final state], [Residual $z_t + hat(Delta z)$],
)

== 9. Config Families Covered Here

The architectures above correspond to the configs currently used in this repository:

- `configs/wildfire/latent_predictor/lstm/model_01_*`
- `configs/wildfire/latent_predictor/transformer/model_01_*`
- `configs/wildfire/latent_predictor/transformer/model_01_residual_*`
- `configs/wildfire/latent_predictor/transformer/model_01_static_*`
- `configs/wildfire/latent_predictor/transformer/model_01_static_head_*`
- `configs/wildfire/latent_predictor/transformer/model_01_static_film_head_*`
- `configs/wildfire/latent_predictor/transformer/model_01_static_film_head_residual_*`
