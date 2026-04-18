#heading(level: 1)[Latent Decoder Worklog]

This document records the main implementation and debugging milestones for Model Two, the latent decoder stage of the wildfire pipeline.

#heading(level: 2)[Purpose]

The goal of Model Two is to reconstruct the next spatial state from latent information and a previous image-like state representation.

In the ground-truth decoder setting, the core contract is:

$
(x_t, z_t, z_(t+1)) -> hat(x)_(t+1)
$

Later pipeline evaluation replaces the target latent with a prediction from Model One.

#heading(level: 2)[Phase 1: Initial Conditional U-Net Decoder]

The first decoder implementation was a bottleneck-conditioned U-Net:

- previous image as spatial input
- previous latent embedding
- target latent embedding
- projected latent maps fused at the bottleneck

Repository path:

- `src/wildfire/scripts/model2_train/15_train_latent_decoder_conditional_unet.py`

This established the basic decoder training and logging loop, but the original image target representation was still tied to the early fire-frame preprocessing assumptions.

#heading(level: 2)[Phase 2: Binary Decoder Variant]

The next step was to move to a single-channel binary-mask formulation:

- input image: one channel
- output image: one channel
- logits output with `BCEWithLogitsLoss`
- hard-threshold metrics added for evaluation

Repository path:

- `src/wildfire/scripts/model2_train/16_train_latent_decoder_conditional_unet_binary.py`

This made the decoder closer to the real segmentation task, but the run behavior still looked weak.

#heading(level: 2)[Critical Debugging Finding]

Inspection of the real decoder targets showed that the masks were not actually loaded as clean binary $\{0, 1\}$ values.

Instead, the observed values were approximately:

- background: $30 / 255 approx 0.1176$
- foreground: $215 / 255 approx 0.8431$

This meant that:

- the target representation was two-level grayscale rather than truly binary
- hard overlap metrics were harder to interpret
- the decoder was not solving exactly the intended task

Inspection script:

- `src/wildfire/scripts/model2_eval/17_inspect_decoder_mask_values.py`

#heading(level: 2)[Phase 3: Corrected Binarized Decoder]

To fix the target mismatch, a corrected binary-mask decoder path was introduced:

- binarize target masks on load with threshold $0.5$
- keep logits output
- evaluate hard Dice / IoU using prediction threshold $0.4$

Repository path:

- `src/wildfire/scripts/model2_train/18_train_latent_decoder_conditional_unet_binary_binarized.py`

This became the real decoder baseline.

The jump in performance after this fix strongly suggested that the target encoding issue had been a dominant source of error.

#heading(level: 2)[Phase 4: Predicted-Latent Pipeline Evaluation]

Once the corrected decoder baseline was established, predicted embeddings from Model One were exported and fed into a decoder evaluation script.

This pipeline evaluation used:

- real previous image
- predicted previous embedding
- predicted target embedding
- real target image

Repository paths:

- latent export: `src/wildfire/scripts/model1_eval/19_export_latent_predictor_embeddings.py`
- predicted-latent decoder evaluation: `src/wildfire/scripts/model2_eval/20_eval_latent_decoder_conditional_unet_binary_predicted.py`

For the one-step teacher-forced pipeline setting, the results remained strong, showing that the selected Model One latent predictor and Model Two decoder interface were compatible.

#heading(level: 2)[Subset Rule in Pipeline Evaluation]

The predicted-latent decoder evaluation only used timesteps where both latent inputs to the decoder were genuinely predicted.

With history length $5$:

- first predicted latent appears once a full 5-step context exists
- decoder evaluation starts at $text(target_idx) = 6$

So the pipeline does not evaluate all sequence timesteps. It evaluates only the subset where the invariant

$
hat(z)_t text( and ) hat(z)_(t+1)
$

both hold.

#heading(level: 2)[Fire-Frame vs Isochrone Decoder Branches]

After the fire-frame decoder line was stabilized, the same decoder logic was reused for isochrone representations.

Because the decoder dataset and trainer already support `input_type`, the ground-truth isochrone decoder did not require a fully separate implementation branch.

Ground-truth isochrone config:

- `configs/wildfire/latent_decoder/conditional_unet_binary/model_01_binarized_05_04_isochrones.toml`

This made it possible to compare:

- fire-frame decoder behavior
- isochrone decoder behavior

under the same corrected binarized training setup.

#heading(level: 2)[Landscape-Conditioned Decoder Branch]

When isochrone reconstructions showed a tendency to close irregular boundaries and fill gaps too uniformly, a new question emerged:

- was the decoder missing local spatial constraints from terrain?

The current U-Net implementation already had standard skip connections, so the next experimental direction was not "add skips" but "add spatial landscape conditioning."

This led to a separate landscape-conditioned decoder branch:

- previous image enters as before
- a spatial landscape patch is concatenated with the image input at the encoder entrance
- latent embeddings are still fused at the bottleneck

Repository path:

- `src/wildfire/scripts/model2_train/22_train_latent_decoder_conditional_unet_landscape_binary.py`

This branch uses:

- raw landscape directory for `indices.json`
- processed landscape raster directory for `landscape_channels_chw.npy`

so that the decoder can consume aligned spatial terrain features directly.

#heading(level: 2)[Current Experimental Structure]

The decoder work is now split into four meaningful lines:

1. fire-frame ground-truth decoder baseline
2. fire-frame predicted-latent pipeline evaluation
3. isochrone ground-truth decoder baseline
4. isochrone landscape-conditioned decoder ablation

This structure keeps the decoder experiments modular:

- target representation debugging is separated from architectural changes
- pipeline evaluation is separated from ground-truth decoder training
- landscape conditioning is treated as a distinct ablation rather than an in-place modification

#heading(level: 2)[Main Lessons So Far]

The most important findings from the decoder worklog are:

- target representation matters as much as architecture
- the corrected binary-mask formulation is much more faithful than the early grayscale-target version
- the Model One to Model Two interface works well in the one-step teacher-forced pipeline setting
- isochrone reconstruction appears harder than fire-frame reconstruction
- when the failure mode is over-smoothing of local fronts, terrain-aware spatial conditioning is a more plausible next step than simply adding skip connections that already exist

#heading(level: 2)[Relevant Files]

- `src/wildfire/data/decoder_dataset.py`
- `src/wildfire/model_latent_decoder/conditional_unet_01.py`
- `src/wildfire/model_latent_decoder/conditional_unet_binary_01.py`
- `src/wildfire/model_latent_decoder/conditional_unet_landscape_binary_01.py`
- `src/wildfire/scripts/model2_train/15_train_latent_decoder_conditional_unet.py`
- `src/wildfire/scripts/model2_train/16_train_latent_decoder_conditional_unet_binary.py`
- `src/wildfire/scripts/model2_train/18_train_latent_decoder_conditional_unet_binary_binarized.py`
- `src/wildfire/scripts/model2_train/22_train_latent_decoder_conditional_unet_landscape_binary.py`
- `src/wildfire/scripts/model2_eval/17_inspect_decoder_mask_values.py`
- `src/wildfire/scripts/model2_eval/20_eval_latent_decoder_conditional_unet_binary_predicted.py`
