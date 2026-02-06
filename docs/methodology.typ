= Methodology

== 1. Problem Setup

Let a wildfire sequence be represented as binary fire masks:

$S_t in {0,1}^{400 times 400}$

The objective is to predict the next fire state:

$hat(S)_{t+1} = F_theta(S_{t-K+1:t}, C_(text("static")), C_(text("dynamic"), t))$

where:

- $K=5$ history frames (standardized 6-frame sequences)
- $C_(text("static"))$: terrain and fuel descriptors
- $C_(text("dynamic"), t)$: time-varying atmospheric drivers (e.g., wind)

== 2. Latent State Representation

Each frame is encoded using a pretrained visual encoder:

$z_t = E_(text("DINO"))(S_t) in R^E$

where $E in {384, 1024}$.

Static drivers:

$g in R^{d_g}$

Dynamic drivers:

$w_t in R^{d_w}$

Input tokens are formed as:

$x_t = [z_t | w_t | g]$

== 3. Model 1 — Latent Temporal Dynamics Predictor

A transformer encoder $T_theta$ models the evolution of latent states:

$H = T_theta(x_{t-4:t})$

The last output token is projected to the next embedding:

$hat(z)_{t+1} = W h_t + b$

=== Loss

$L_(text("latent")) = (z_{t+1} - hat(z)_{t+1})^2$

== 4. Model 2 — Conditional Spatial Decoder

The predicted embedding is translated to pixel space:

$hat(S)_{t+1} = D(S_t, z_t, hat(z)_{t+1})$

The decoder is a conditional U-Net where embeddings are projected to spatial feature maps and fused with encoded image features.

=== Pixel Loss

$L_(text("pix")) = text("BCE")(S_{t+1}, hat(S)_{t+1}) + lambda (1 - text("Dice"))$

== 5. Training Procedure

1. Precompute DINO embeddings for all frames.
2. Train Model 1 to predict latent embeddings.
3. Train Model 2 to reconstruct pixel masks.
4. Optionally fine-tune end-to-end.

== 6. Evaluation Metrics

- IoU / Jaccard
- Dice coefficient
- Precision–Recall AUC
- Boundary F1-score

== 7. Ablation Studies

=== Temporal Model

- LSTM vs Transformer

=== Decoder Conditioning

- $D(S_t, hat(z)_{t+1})$
- $D(S_t, z_t, hat(z)_{t+1})$
- $D(S_t, z_{t-1}, z_t, hat(z)_{t+1})$

=== Environmental Conditioning

- With vs without wind
- With vs without terrain

== 8. Output

Final prediction:

$hat(S)_{t+1} in [0,1]^{400 times 400}$

Thresholding is applied for binary evaluation.
