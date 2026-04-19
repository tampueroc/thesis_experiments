= Architectures Used In The Wildfire Pipeline

This document summarizes the main architectures used in this repository for wildfire spread forecasting.

The pipeline has two learned stages:

- Model One: latent temporal prediction in embedding space
- Model Two: latent-conditioned spatial decoding back to the next wildfire image

The first half of this document describes the latent temporal predictors. The second half describes the decoder families used to reconstruct the next image from latent states.

== Part I: Model One Latent Predictors

=== Shared Setup

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

=== 1. LSTM Predictor

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

=== 2. Plain Transformer Predictor

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

=== 3. Residual Transformer Predictor

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

=== 4. Static-Conditioned Transformer

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

=== 5. Static-Head Transformer

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

=== 6. Static FiLM-Head Transformer

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

=== 7. Static FiLM-Head Residual Transformer

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

=== 8. Predictor Summary

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

=== 9. Predictor Config Families Covered Here

The architectures above correspond to the configs currently used in this repository:

- `configs/wildfire/latent_predictor/lstm/model_01_*`
- `configs/wildfire/latent_predictor/transformer/model_01_*`
- `configs/wildfire/latent_predictor/transformer/model_01_residual_*`
- `configs/wildfire/latent_predictor/transformer/model_01_static_*`
- `configs/wildfire/latent_predictor/transformer/model_01_static_head_*`
- `configs/wildfire/latent_predictor/transformer/model_01_static_film_head_*`
- `configs/wildfire/latent_predictor/transformer/model_01_static_film_head_residual_*`

== Part II: Model Two Latent Decoders

=== Shared Decoder Contract

In the ground-truth decoder setting, each sample uses:

- previous image: $x_t$
- previous embedding: $z_t$
- target embedding: $z_(t+1)$
- target image: $x_(t+1)$

The decoder predicts the next image from the previous image together with latent conditioning:

$hat(x)_(t+1) = f_(theta)(x_t, z_t, z_(t+1))$

For the corrected binary-mask runs, the target image is binarized at load time using a target threshold of $0.5$. Reported hard metrics are computed from thresholded decoder probabilities using a prediction threshold of $0.4$.

For end-to-end evaluation, the target latent can be replaced by a predicted latent from Model One. In the stronger predicted-latent evaluation setting used here, both latent inputs to the decoder are predicted:

$hat(x)_(t+1) = f_(theta)(x_t, hat(z)_t, hat(z)_(t+1))$

=== 10. Initial Conditional U-Net Decoder

Files:

- `src/wildfire/model_latent_decoder/conditional_unet_01.py`
- `src/wildfire/scripts/model2_train/15_train_latent_decoder_conditional_unet.py`

The initial decoder is a bottleneck-conditioned U-Net. The previous image is encoded through a convolutional encoder, the two latent embeddings are projected and fused at the bottleneck, and the result is decoded back to image space with skip connections.

$e = text("Encoder")(x_t)$

$c = phi([z_t | z_(t+1)])$

$b = psi([e_(text("bottleneck")) | c])$

$hat(x)_(t+1) = text("Decoder")(b, text("skips"))$

Key characteristics:

- standard U-Net encoder/decoder with skip connections
- latent conditioning injected at the bottleneck
- original early baseline before the corrected binary-mask target path

=== 11. Binary Conditional U-Net Decoder

Files:

- `src/wildfire/model_latent_decoder/conditional_unet_binary_01.py`
- `src/wildfire/scripts/model2_train/16_train_latent_decoder_conditional_unet_binary.py`
- `src/wildfire/scripts/model2_train/18_train_latent_decoder_conditional_unet_binary_binarized.py`

This branch adapts the decoder to one-channel binary-mask prediction. The corrected `script 18` path is the real ground-truth decoder baseline because it uses true binary targets rather than two-level grayscale mask values.

$ell = f_(theta)(x_t, z_t, z_(t+1))$

$p_(t+1) = sigma(ell)$

where $ell$ are output logits and $sigma$ is the sigmoid nonlinearity used for evaluation and soft losses.

Key characteristics:

- one-channel input/output decoder
- bottleneck latent conditioning
- trained with BCE-with-logits plus soft Dice
- hard Dice and hard IoU computed after thresholding predicted probabilities
- reused for both fire-frame and isochrone representations through config changes

=== 12. Predicted-Latent Decoder Evaluation

Files:

- `src/wildfire/scripts/model1_eval/19_export_latent_predictor_embeddings.py`
- `src/wildfire/scripts/model2_eval/20_eval_latent_decoder_conditional_unet_binary_predicted.py`

This is the end-to-end system benchmark rather than a separate trained decoder family. A selected Model One predictor exports latent sequences, and the best trained decoder checkpoint is evaluated using predicted latent inputs instead of ground-truth future latents.

For the teacher-forced one-step benchmark used here:

- Model One predicts $hat(z)_(t+1)$ from a 5-step latent history
- decoder evaluation only starts once both $hat(z)_t$ and $hat(z)_(t+1)$ are genuinely predicted
- this corresponds to $text("target_idx") >= 6$

The decoder contract becomes:

$hat(x)_(t+1) = f_(theta)(x_t, hat(z)_t, hat(z)_(t+1))$

Key characteristics:

- no decoder retraining
- same decoder weights as the selected ground-truth decoder baseline
- used as the final end-to-end benchmark

=== 13. Landscape-Conditioned Isochrone Decoder

Files:

- `src/wildfire/model_latent_decoder/conditional_unet_landscape_binary_01.py`
- `src/wildfire/scripts/model2_train/22_train_latent_decoder_conditional_unet_landscape_binary.py`

This branch augments the binary conditional decoder with spatial landscape information. A per-sequence landscape patch is cropped from a processed raster cube, resized to the image resolution, and concatenated with the previous image at the encoder input.

$x'_t = [x_t | l_t]$

$hat(x)_(t+1) = f_(theta)(x'_t, z_t, z_(t+1))$

The current stabilized version does not use the full raw 8-channel landscape stack. Instead, it uses a normalized continuous subset:

- canopy bulk density (`cbd`)
- canopy base height (`cbh`)
- elevation

Normalization statistics are computed on the training split only and reused for validation.

Key characteristics:

- spatial landscape conditioning at image resolution
- raw and processed landscape paths separated:
  - raw folder provides `indices.json`
  - processed folder provides `landscape_channels_chw.npy`
- current landscape ablation is specific to the isochrone decoder line

=== 14. Decoder Summary

#table(
  columns: 4,
  [Variant], [Image Inputs], [Latent Inputs], [Notes],
  [Initial Conditional U-Net], [Previous image], [$z_t, z_(t+1)$], [Original 3-channel-style baseline],
  [Binary Conditional U-Net], [Previous image], [$z_t, z_(t+1)$], [Corrected one-channel binary decoder baseline],
  [Predicted-Latent Evaluation], [Previous image], [$hat(z)_t, hat(z)_(t+1)$], [End-to-end benchmark with fixed decoder checkpoint],
  [Landscape Isochrone Decoder], [Previous image + landscape patch], [$z_t, z_(t+1)$], [Isochrone decoder ablation with normalized continuous terrain channels],
)

=== 15. Decoder Config Families Covered Here

- `configs/wildfire/latent_decoder/conditional_unet/model_01_*`
- `configs/wildfire/latent_decoder/conditional_unet_binary/model_01_*`
- `configs/wildfire/latent_decoder/conditional_unet_landscape_binary/model_01_*`
