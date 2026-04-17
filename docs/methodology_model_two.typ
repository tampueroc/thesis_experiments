#heading(level: 1)[Model Two - Conditional U-Net Decoder]

Model Two reconstructs the next wildfire frame in pixel space from:

- the previous frame $x_t$
- the previous latent embedding $z_t$
- the target latent embedding $z_(t+1)$

Its role in the pipeline is to map latent wildfire state transitions back into a spatial fire map.

#heading(level: 2)[Implemented Input Contract]

In the current repository implementation, each decoder sample is

$
(x_t, z_t, z_(t+1)) -> hat(x)_(t+1)
$

with:

- $x_t in RR^(3 times 400 times 400)$
- $z_t in RR^E$
- $z_(t+1) in RR^E$
- $hat(x)_(t+1) in RR^(3 times 400 times 400)$

The image tensors are built from the stored wildfire masks by converting each source image to grayscale, scaling to $[0, 1]$, and repeating the single channel three times. This keeps the decoder image representation aligned with the representation used during DINO embedding extraction.

#heading(level: 2)[Dataset Construction]

The decoder dataset is built from the embedding manifest and the original frame tree.

For each wildfire sequence:

1. load the per-sequence embedding array
2. locate the original frame directory from `manifest.json`
3. align frame indices and embedding timesteps
4. emit one sample for each adjacent pair of timesteps

So for a sequence with timesteps $0, 1, ..., T-1$, the dataset emits:

$
(x_0, z_0, z_1) -> x_1
$

$
(x_1, z_1, z_2) -> x_2
$

$
...
$

$
(x_(T-2), z_(T-2), z_(T-1)) -> x_(T-1)
$

If embedding normalization is enabled, each timestep embedding is L2-normalized independently before training, matching the normalization option used in Model One.

#heading(level: 2)[Conditioning Strategy]

The decoder uses late spatial conditioning at the U-Net bottleneck.

Each embedding is projected independently:

$
z_t -> text("Linear") -> RR^(C_c times H_b times W_b)
$

$
z_(t+1) -> text("Linear") -> RR^(C_c times H_b times W_b)
$

where:

- $C_c$ is the conditioning channel count per embedding
- $(H_b, W_b)$ is the bottleneck spatial size

Each projected vector is reshaped into a spatial conditioning map:

$
z_t^* in RR^(C_c times H_b times W_b)
$

$
z_(t+1)^* in RR^(C_c times H_b times W_b)
$

The two conditioning maps are concatenated with the encoder bottleneck features:

$
f_b = text("concat")(h_b, z_t^*, z_(t+1)^*)
$

This fused tensor is then processed by an additional bottleneck convolution block before decoding.

#heading(level: 2)[U-Net Architecture]

The implemented decoder is a standard convolutional U-Net with four downsampling stages in the base configuration.

== Encoder

Each encoder block applies:

$
text("Conv 3x3") -> text("ReLU") -> text("Conv 3x3") -> text("ReLU")
$

followed by:

$
text("MaxPool 2x2")
$

At each resolution, the pre-pooled feature map is stored as a skip connection.

If the base channel count is $C$, the encoder channel progression is:

$
C, 2C, 4C, 8C
$

and the bottleneck channel width is:

$
16C
$

For the default implementation:

- base channels = $32$
- encoder widths = $32, 64, 128, 256$
- bottleneck width = $512$

== Bottleneck

The image stream first passes through a bottleneck convolution block:

$
h_b = phi_(text("bottleneck"))(h_(text("enc")))
$

Then latent conditioning is fused:

$
tilde(h)_b = phi_(text("fused"))([h_b | z_t^* | z_(t+1)^*])
$

In the default configuration:

- bottleneck spatial size = $25 times 25$
- conditioning channels per embedding = $32$
- total conditioning width = $64$

== Decoder

Each decoder stage applies:

$
text("ConvTranspose 2x2") -> text("concat with skip") -> text("Conv 3x3") -> text("ReLU") -> text("Conv 3x3") -> text("ReLU")
$

The channel progression mirrors the encoder in reverse until the spatial resolution returns to $400 times 400$.

== Output Head

The output head is a $1 times 1$ convolution:

$
hat(y) = W_(text("out")) h_(text("dec"))
$

producing:

$
hat(y) in RR^(3 times 400 times 400)
$

During training and evaluation, these output logits are passed through a sigmoid when computing soft overlap metrics or when rendering qualitative reconstructions.

#heading(level: 2)[Training Objective]

The current baseline is trained with a hybrid reconstruction objective:

$
cal(L) = cal(L)_(text("BCEWithLogits")) + lambda_(text("dice")) cal(L)_(text("Dice"))
$

where the default decoder configuration uses:

$
lambda_(text("dice")) = 1.0
$

The binary cross-entropy term supervises pixel-wise reconstruction, and the Dice term encourages overlap quality on the sparse fire-mask structure.

#heading(level: 2)[Evaluation Metrics]

The decoder trainer logs:

- total loss
- BCE loss
- Dice loss
- Dice coefficient
- IoU

for train and validation splits, and final metrics for the holdout split.

The trainer also saves qualitative reconstruction panels showing:

- previous frame
- predicted next frame
- target next frame

for validation and holdout examples.

#heading(level: 2)[Relationship To Model One]

Model One and Model Two are intentionally separated:

- Model One learns temporal dynamics in latent space
- Model Two learns spatial reconstruction in image space

During baseline decoder training, Model Two uses ground-truth latent targets:

$
(x_t, z_t, z_(t+1)) -> x_(t+1)
$

This isolates decoder quality from temporal prediction error.

For full pipeline evaluation, the target latent can later be replaced by a predicted latent from Model One:

$
(x_t, z_t, hat(z)_(t+1)) -> hat(x)_(t+1)
$

#heading(level: 2)[Repository Files]

The current Model Two implementation is defined by:

- `src/wildfire/data/decoder_dataset.py`
- `src/wildfire/model_latent_decoder/conditional_unet_01.py`
- `src/wildfire/scripts/model2_train/15_train_latent_decoder_conditional_unet.py`
- `configs/wildfire/latent_decoder/conditional_unet/model_01_base.toml`

#heading(level: 2)[Summary]

The implemented decoder is a bottleneck-conditioned U-Net that combines:

- spatial context from the previous wildfire frame
- latent state information from the previous embedding
- latent target information from the next embedding

to reconstruct the next fire frame.

In compact form:

$
hat(x)_(t+1) = f_(theta)(x_t, z_t, z_(t+1))
$

This makes Model Two the spatial reconstruction component of the wildfire forecasting pipeline.
