# Scratchpad

This file is the persistent decision log for the byte-level pure-JEPA workspace.

## Defaults

- Dataset family: `byte260`
- Track: `non-record`
- Skeleton owner: `train_jepa.py`
- Main experimentation surface: `model.py`
- Comparator baseline: `train_baseline.py`
- Score policy: log exact challenge `val_bpb` only when the model exposes `nll_sum_nat`; otherwise label results as proxy-only
- Special-token policy: control ids stay in context, but are excluded from JEPA target loss and exact byte-level scoring

## Experiment Log


| Date       | Run ID                                                                | Hypothesis                                                                                                                      | Key Config                                                                                                                                                                                                                                                                                                                                                                                                                                | Proxy Val | Exact val_bpb | Artifact Bytes | Decision | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ---------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ------------- | -------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-03-25 | scaffold                                                              | Initial byte-level pure-JEPA workspace, copied trainer skeleton, dummy model only                                               | `byte260`, proxy validation only, int8 export enabled                                                                                                                                                                                                                                                                                                                                                                                     | N/A       | N/A           | N/A            | keep     | First runnable scaffold. No scoring bridge yet.                                                                                                                                                                                                                                                                                                                                                                                                                            |
| 2026-03-25 | baseline_scaffold                                                     | Add a byte-level GPT comparator close to the root naive baseline                                                                | `byte260`, exact `val_bpb`, control targets excluded from byte score                                                                                                                                                                                                                                                                                                                                                                      | N/A       | N/A           | N/A            | keep     | Comparison-only path in `train_baseline.py`; does not change the JEPA objective.                                                                                                                                                                                                                                                                                                                                                                                           |
| 2026-03-25 | smoke_scale_sweep_base                                                | Scaled cosine logits may remove the bounded-logit floor without changing training                                               | `120 steps`, `16 MiB` val subset, `ema` target table, post-hoc scale sweep                                                                                                                                                                                                                                                                                                                                                                | `1.3295`  | N/A           | `2,561,088`    | discard  | Subset-only smoke. Best post-hoc subset bpb was `5.5837` at scale `8.0`; simple calibration was not enough.                                                                                                                                                                                                                                                                                                                                                                |
| 2026-03-25 | smoke_mse_dot4                                                        | Magnitude-aware latent regression plus raw-dot bridge may outperform cosine alignment                                           | `120 steps`, `16 MiB` val subset, `JEPA_LOSS=mse`, `EVAL_BRIDGE=dot`, `scale=4`                                                                                                                                                                                                                                                                                                                                                           | `0.0005`  | N/A           | `2,549,315`    | discard  | Subset-only smoke. Regression loss collapsed but subset bpb stayed at `7.9904`; matching magnitudes alone did not improve byte discrimination.                                                                                                                                                                                                                                                                                                                             |
| 2026-03-25 | smoke_fixed_random_cos8                                               | A frozen normalized random byte codebook may provide cleaner target geometry than the EMA token table                           | `120 steps`, `16 MiB` val subset, `TARGET_CODEBOOK=fixed_random`, `scale=8`                                                                                                                                                                                                                                                                                                                                                               | `1.3042`  | N/A           | `2,575,916`    | keep     | Subset-only smoke. Reached `5.4037` subset bpb; post-hoc sweep over scales `4/6/8/10/12/16` confirmed `8.0` as best.                                                                                                                                                                                                                                                                                                                                                       |
| 2026-03-25 | smoke_fixed_random_cos8_400                                           | The fixed-codebook direction should keep improving with a longer smoke run                                                      | `400 steps`, `16 MiB` val subset, `TARGET_CODEBOOK=fixed_random`, `scale=8`                                                                                                                                                                                                                                                                                                                                                               | `1.2710`  | N/A           | `2,569,083`    | keep     | Subset-only smoke. Best JEPA result so far at `5.3192` subset bpb, but still far from the `2.5` milestone.                                                                                                                                                                                                                                                                                                                                                                 |
| 2026-03-25 | smoke_fixed_hadamard_cos8                                             | An orthogonal Hadamard payload codebook may beat the random fixed codebook                                                      | `120 steps`, `16 MiB` val subset, `TARGET_CODEBOOK=fixed_hadamard`, `scale=8`                                                                                                                                                                                                                                                                                                                                                             | `1.2995`  | N/A           | `2,535,919`    | keep     | Subset-only smoke. Slightly better than single random codebook at the same 120-step budget with `5.3444` subset bpb, but still behind the 400-step random run.                                                                                                                                                                                                                                                                                                             |
| 2026-03-25 | smoke_fixed_random2_cos4                                              | Two independent fixed random codebooks may give a sharper error-correcting target space                                         | `120 steps`, `16 MiB` val subset, `NUM_TARGET_CODEBOOKS=2`, `TARGET_CODEBOOK=fixed_random`, `scale=4`                                                                                                                                                                                                                                                                                                                                     | `1.3060`  | N/A           | `2,720,474`    | discard  | Subset-only smoke. Multi-codebook logits did not beat the single-codebook best; scale sweep over `2/3/4/5/6` still bottomed out at `5.4166`.                                                                                                                                                                                                                                                                                                                               |
| 2026-03-25 | smoke_big_hadamard384                                                 | The fixed-Hadamard branch may be capacity-limited rather than concept-limited                                                   | `100 steps`, `MODEL_DIM=384`, `6 layers`, `16 MiB` val subset, `TARGET_CODEBOOK=fixed_hadamard`                                                                                                                                                                                                                                                                                                                                           | `1.3088`  | N/A           | `6,185,258`    | discard  | Subset-only smoke. Larger capacity increased artifact size and runtime, but did not improve early bpb; final subset bpb was `5.4257`.                                                                                                                                                                                                                                                                                                                                      |
| 2026-03-25 | smoke_fixed_random_lr1e3                                              | The best small fixed-codebook branch may just need more aggressive optimization                                                 | `120 steps`, `LR=1e-3`, `WEIGHT_DECAY=0`, `16 MiB` val subset, `TARGET_CODEBOOK=fixed_random`                                                                                                                                                                                                                                                                                                                                             | `1.3212`  | N/A           | `2,644,835`    | discard  | Subset-only smoke. Higher LR hurt both proxy loss and subset bpb; final subset bpb was `5.6284`.                                                                                                                                                                                                                                                                                                                                                                           |
| 2026-03-25 | smoke_contextual_random75                                             | A bidirectional EMA target encoder mixed with the anchor codebook may improve representations enough to help retrieval          | `120 steps`, `CONTEXTUAL_TARGET=1`, `TARGET_ANCHOR_WEIGHT=0.75`, `PROTOTYPE_EMA_DECAY=0.95`, `fixed_random` anchor                                                                                                                                                                                                                                                                                                                        | `0.0351`  | N/A           | `4,970,609`    | discard  | Subset-only smoke. Latent loss collapsed, but both anchor-table and prototype-table bridges stayed near `7.93-7.97` subset bpb. This contextual target path broke token discrimination.                                                                                                                                                                                                                                                                                    |
| 2026-03-25 | smoke_fixed_random_vicreg                                             | VICReg-style variance/covariance regularization may improve the static fixed-codebook representation geometry                   | `120 steps`, `fixed_random`, `VAR_REG_WEIGHT=0.1`, `COV_REG_WEIGHT=0.01`, `scale=8`                                                                                                                                                                                                                                                                                                                                                       | `1.3140`  | N/A           | `2,662,710`    | discard  | Subset-only smoke. Regularization did not help; final subset bpb was `5.4925`, worse than the unregularized fixed-random control.                                                                                                                                                                                                                                                                                                                                          |
| 2026-03-25 | smoke_contextual_resid010                                             | An orthogonal contextual residual around a fixed random anchor may preserve token identity better than free contextual blending | `120 steps`, `fixed_random`, `CONTEXTUAL_TARGET=1`, `CONTEXTUAL_RESIDUAL_SCALE=0.1`, `scale=8`                                                                                                                                                                                                                                                                                                                                            | `1.2931`  | N/A           | `4,985,395`    | discard  | Subset-only smoke. This repaired the catastrophic contextual failure and held `5.4860` subset bpb after int8+zlib roundtrip, but it still trailed the static fixed-random control.                                                                                                                                                                                                                                                                                         |
| 2026-03-25 | smoke_anchor_aux025                                                   | Keep the scored anchor target unchanged and add a JEPA-only contextual auxiliary head during training                           | `120 steps`, `fixed_random`, `CONTEXTUAL_AUX_WEIGHT=0.25`, `scale=8`                                                                                                                                                                                                                                                                                                                                                                      | `1.3115`  | N/A           | `5,132,339`    | discard  | Subset-only smoke. The decoupled context head trained cleanly but finished at `5.4901` subset bpb, no better than the simpler residual branch.                                                                                                                                                                                                                                                                                                                             |
| 2026-03-25 | smoke_hadamard_resid010                                               | An orthogonal contextual residual may pair better with the sharper Hadamard anchor geometry than with the random anchor         | `120 steps`, `fixed_hadamard`, `CONTEXTUAL_TARGET=1`, `CONTEXTUAL_RESIDUAL_SCALE=0.1`, `scale=8`                                                                                                                                                                                                                                                                                                                                          | `1.2688`  | N/A           | `4,878,603`    | keep     | Subset-only smoke. Early result reached `5.3487` subset bpb after int8+zlib roundtrip, essentially matching the plain Hadamard control and justifying a longer run.                                                                                                                                                                                                                                                                                                        |
| 2026-03-25 | smoke_hadamard_resid010_400                                           | The Hadamard residual branch may need more budget before the contextual signal pays off                                         | `400 steps`, `fixed_hadamard`, `CONTEXTUAL_TARGET=1`, `CONTEXTUAL_RESIDUAL_SCALE=0.1`, `scale=8`                                                                                                                                                                                                                                                                                                                                          | `1.2202`  | N/A           | `4,890,361`    | keep     | Subset-only smoke. New best JEPA result so far at `5.1006` subset bpb after int8+zlib roundtrip, beating the old `5.3192` control. Post-hoc scale sweeps over `4/6/7/7.5/8/8.5/9/10/12/16` confirmed `8.0` as the best bridge scale for this checkpoint.                                                                                                                                                                                                                   |
| 2026-03-25 | smoke_hadamard_resid010_ema995_400                                    | A slower target EMA may stabilize the contextual residual target enough to improve the late-run Hadamard branch                 | `400 steps`, `fixed_hadamard`, `TARGET_EMA_DECAY=0.995`, `CONTEXTUAL_TARGET=1`, `CONTEXTUAL_RESIDUAL_SCALE=0.1`, `scale=8`                                                                                                                                                                                                                                                                                                                | `1.2160`  | N/A           | `4,880,890`    | discard  | Subset-only smoke. Slower EMA improved the mid-run checkpoints but finished at `5.1299` subset bpb after int8+zlib roundtrip, slightly behind the `0.99` EMA control.                                                                                                                                                                                                                                                                                                      |
| 2026-03-26 | smoke_hadamard_resid010_mse_energy_400                                | Matching both the JEPA loss and the softmax energy to MSE may improve the Hadamard residual branch                              | `400 steps`, `16 MiB` val subset, `TARGET_CODEBOOK=fixed_hadamard`, `CONTEXTUAL_TARGET=1`, `CONTEXTUAL_RESIDUAL_SCALE=0.1`, `JEPA_LOSS=mse`, `EVAL_BRIDGE=mse`, train `scale=8`, post-hoc best `scale=2048`                                                                                                                                                                                                                               | `0.0034`  | N/A           | `4,810,133`    | discard  | Subset-only smoke. The raw run at `scale=8` was almost uniform at `8.0100` subset bpb; a post-hoc scale sweep improved it to `5.7481` at `2048`, but it still trailed the cosine control `smoke_hadamard_resid010_400` (`5.1006`).                                                                                                                                                                                                                                         |
| 2026-03-26 | smoke_hadamard_resid010_cosine_lmprobe1_400                           | A detached LM probe on the predicted JEPA latent may reveal next-byte distribution information without perturbing JEPA training | `400 steps`, `16 MiB` val subset, `TARGET_CODEBOOK=fixed_hadamard`, `CONTEXTUAL_TARGET=1`, `CONTEXTUAL_RESIDUAL_SCALE=0.1`, `JEPA_LOSS=cosine`, `LM_PROBE_WEIGHT=1`                                                                                                                                                                                                                                                                       | `1.2110`  | N/A           | `4,883,239`    | keep     | Subset-only smoke. User-approved hybrid/probe branch, not pure-JEPA strict. The detached LM head reached `3.6791` subset bpb after int8+zlib roundtrip, far better than the pure-JEPA cosine bridge control `smoke_hadamard_resid010_400` (`5.1006`), while the JEPA proxy stayed comparable (`1.2110` vs `1.2202`). This points to a bridge/decoder bottleneck more than a representation bottleneck.                                                                     |
| 2026-03-26 | smoke_hadamard_resid010_cosine_lmprobe1_splitopt_400                  | Separate optimizer state and gradient clipping may help the detached LM probe branch                                            | `400 steps`, `16 MiB` val subset, `TARGET_CODEBOOK=fixed_hadamard`, `CONTEXTUAL_TARGET=1`, `CONTEXTUAL_RESIDUAL_SCALE=0.1`, `JEPA_LOSS=cosine`, `LM_PROBE_WEIGHT=1`, `SPLIT_LM_PROBE_OPTIMIZER=1`, `LM_PROBE_LR=3e-4`                                                                                                                                                                                                                     | `1.2216`  | N/A           | `4,880,166`    | discard  | Subset-only smoke. User-approved hybrid/probe branch, not pure-JEPA strict. Splitting the LM head into its own optimizer and clip path nearly matched but did not beat the simpler shared-optimizer run: `3.6997` subset bpb after int8+zlib roundtrip versus `3.6791` for `smoke_hadamard_resid010_cosine_lmprobe1_400`. This suggests optimizer competition is not the main bottleneck.                                                                                  |
| 2026-03-26 | smoke_hadamard_resid010_cosine_lmprobe1_muon_400                      | The hybrid JEPA+probe branch may benefit from the repo's Muon optimizer grouping                                                | `400 steps`, `16 MiB` val subset, `TARGET_CODEBOOK=fixed_hadamard`, `CONTEXTUAL_TARGET=1`, `CONTEXTUAL_RESIDUAL_SCALE=0.1`, `JEPA_LOSS=cosine`, `LM_PROBE_WEIGHT=1`, `OPTIMIZER=muon`, `EMBED/MATRIX/SCALAR_LR=3e-4`                                                                                                                                                                                                                      | `1.3104`  | N/A           | `4,808,983`    | discard  | Subset-only smoke. User-approved hybrid/probe branch, not pure-JEPA strict. A direct Muon swap at the old AdamW LR scale underperformed badly: `6.8911` subset bpb after int8+zlib roundtrip versus `3.6791` for the AdamW probe run. This does not rule Muon out, but it does show that a naive drop-in replacement is not competitive and would need a dedicated LR retune.                                                                                              |
| 2026-03-26 | **smoke_hadamard_resid010_cosine_lmprobe1_muon_tuned_400**            | **Tuned Muon hyperparameters may make the hybrid JEPA+probe branch substantially stronger than AdamW**                          | `400 steps`**,** `16 MiB` **val subset,** `TARGET_CODEBOOK=fixed_hadamard`**,** `CONTEXTUAL_TARGET=1`**,** `CONTEXTUAL_RESIDUAL_SCALE=0.1`, `JEPA_LOSS=cosine`, `LM_PROBE_WEIGHT=1`, `OPTIMIZER=muon`, `MUON_MOMENTUM=0.99`, warmup `0.92->0.99/1500`, `EMBED_LR=0.03`, `MATRIX_LR=0.02`, `SCALAR_LR=0.02`, `GRAD_CLIP_NORM=0.3`, `WARMDOWN_ITERS=3000`                                                                                   | `0.6937`  | N/A           | `5,579,695`    | keep     | Subset-only smoke. User-approved hybrid/probe branch, not pure-JEPA strict. This tuned Muon run reached `2.2588` subset bpb after int8+zlib roundtrip, a large improvement over the AdamW hybrid control `smoke_hadamard_resid010_cosine_lmprobe1_400` (`3.6791`). The prior naive Muon swap failed, so the gain appears to come from the optimizer retune rather than Muon alone.                                                                                         |
| 2026-03-26 | smoke_hadamard_resid010_cosine_lmprobe1_muon_tuned_control_120_ga4    | Reproduce the tuned hybrid branch under a smaller microbatch schedule to create a fair control for larger JEPA variants         | `120 steps`, `16 MiB` val subset, `TARGET_CODEBOOK=fixed_hadamard`, `CONTEXTUAL_TARGET=1`, `CONTEXTUAL_RESIDUAL_SCALE=0.1`, `JEPA_LOSS=cosine`, `LM_PROBE_WEIGHT=1`, `OPTIMIZER=muon`, `MUON_MOMENTUM=0.99`, warmup `0.92->0.99/1500`, `EMBED_LR=0.03`, `MATRIX_LR=0.02`, `SCALAR_LR=0.02`, `GRAD_CLIP_NORM=0.3`, `WARMDOWN_ITERS=3000`, `GRAD_ACCUM_STEPS=4`, `VAL_BATCH_SIZE=131072`                                                    | `1.2546`  | N/A           | `5,086,920`    | keep     | Subset-only smoke. User-approved hybrid/probe branch, not pure-JEPA strict. The smaller microbatch schedule remained strong at `3.8545` subset bpb after int8+zlib roundtrip, giving a stable comparison point for architectural changes that do not fit the default single-GPU microbatch.                                                                                                                                                                                |
| 2026-03-26 | smoke_hadamard_resid010_cosine_lmprobe1_muon_tuned_predstack2_120_ga4 | A lightweight causal predictor stack may decode future JEPA latents better than the plain MLP predictor                         | `120 steps`, `16 MiB` val subset, `TARGET_CODEBOOK=fixed_hadamard`, `CONTEXTUAL_TARGET=1`, `CONTEXTUAL_RESIDUAL_SCALE=0.1`, `JEPA_LOSS=cosine`, `LM_PROBE_WEIGHT=1`, `OPTIMIZER=muon`, `MUON_MOMENTUM=0.99`, warmup `0.92->0.99/1500`, `EMBED_LR=0.03`, `MATRIX_LR=0.02`, `SCALAR_LR=0.02`, `GRAD_CLIP_NORM=0.3`, `WARMDOWN_ITERS=3000`, `PREDICTOR_NUM_LAYERS=2`, `PREDICTOR_NUM_HEADS=4`, `GRAD_ACCUM_STEPS=4`, `VAL_BATCH_SIZE=131072` | `1.2570`  | N/A           | `5,834,728`    | discard  | Subset-only smoke. User-approved hybrid/probe branch, not pure-JEPA strict. The 2-block causal predictor stack finished slightly worse than the matched simple-predictor control (`3.9426` vs `3.8545` subset bpb after int8+zlib) while adding about `0.75 MB` to the artifact and requiring the smaller GA4 microbatch schedule to avoid the default single-GPU OOM.                                                                                                     |
| 2026-03-26 | smoke_hadamard_resid010_cosine_lmprobe1_muon_tuned_5layer_120_ga4     | A deeper JEPA encoder may improve the frozen-latent LM probe more efficiently than extra predictor depth                        | `120 steps`, `16 MiB` val subset, `TARGET_CODEBOOK=fixed_hadamard`, `CONTEXTUAL_TARGET=1`, `CONTEXTUAL_RESIDUAL_SCALE=0.1`, `JEPA_LOSS=cosine`, `LM_PROBE_WEIGHT=1`, `OPTIMIZER=muon`, `MUON_MOMENTUM=0.99`, warmup `0.92->0.99/1500`, `EMBED_LR=0.03`, `MATRIX_LR=0.02`, `SCALAR_LR=0.02`, `GRAD_CLIP_NORM=0.3`, `WARMDOWN_ITERS=3000`, `NUM_LAYERS=5`, `GRAD_ACCUM_STEPS=4`, `VAL_BATCH_SIZE=131072`                                    | `1.2686`  | N/A           | `6,094,580`    | discard  | Subset-only smoke. User-approved hybrid/probe branch, not pure-JEPA strict. The naive `5`-layer scale-up also trailed the matched `4`-layer control (`3.9403` vs `3.8545` subset bpb after int8+zlib) while adding about `1.0 MB` to the artifact, so simple early-budget depth scaling is not yet paying off under the current optimizer settings.                                                                                                                        |
| 2026-03-26 | smoke_hadamard_resid010_cosine_lmprobe1_muon_tuned_control_400_ga4    | Check whether the GA4 control reproduces the tuned 400-step hybrid quality closely enough for matched scale-up comparisons      | `400 steps`, `16 MiB` val subset, `TARGET_CODEBOOK=fixed_hadamard`, `CONTEXTUAL_TARGET=1`, `CONTEXTUAL_RESIDUAL_SCALE=0.1`, `JEPA_LOSS=cosine`, `LM_PROBE_WEIGHT=1`, `OPTIMIZER=muon`, `MUON_MOMENTUM=0.99`, warmup `0.92->0.99/1500`, `EMBED_LR=0.03`, `MATRIX_LR=0.02`, `SCALAR_LR=0.02`, `GRAD_CLIP_NORM=0.3`, `WARMDOWN_ITERS=3000`, `GRAD_ACCUM_STEPS=4`, `VAL_BATCH_SIZE=131072`                                                    | `0.7289`  | N/A           | `5,399,962`    | keep     | Subset-only smoke. User-approved hybrid/probe branch, not pure-JEPA strict. The matched GA4 control reached `2.3568` subset bpb after int8+zlib roundtrip, slightly worse than the earlier `2.2588` tuned run but close enough to serve as the fair 400-step comparison baseline for capacity changes that need smaller microbatches.                                                                                                                                      |
| 2026-03-26 | smoke_hadamard_resid010_cosine_lmprobe1_muon_tuned_5layer_400_ga4     | The 5-layer JEPA encoder may need the full 400-step budget before its extra capacity helps the detached LM probe                | `400 steps`, `16 MiB` val subset, `TARGET_CODEBOOK=fixed_hadamard`, `CONTEXTUAL_TARGET=1`, `CONTEXTUAL_RESIDUAL_SCALE=0.1`, `JEPA_LOSS=cosine`, `LM_PROBE_WEIGHT=1`, `OPTIMIZER=muon`, `MUON_MOMENTUM=0.99`, warmup `0.92->0.99/1500`, `EMBED_LR=0.03`, `MATRIX_LR=0.02`, `SCALAR_LR=0.02`, `GRAD_CLIP_NORM=0.3`, `WARMDOWN_ITERS=3000`, `NUM_LAYERS=5`, `GRAD_ACCUM_STEPS=4`, `VAL_BATCH_SIZE=131072`                                    | `0.7406`  | N/A           | `6,442,735`    | discard  | Subset-only smoke. User-approved hybrid/probe branch, not pure-JEPA strict. Even at the full `400`-step budget, the naive `5`-layer scale-up still trailed the matched `4`-layer GA4 control (`2.3872` vs `2.3568` subset bpb after int8+zlib) while taking about `22%` longer per step and adding about `1.04 MB` to the artifact. This suggests the extra depth is not simply undertrained; under the current optimizer and architecture it is a worse efficiency trade. |


## Current Controls

- Strict pure-JEPA control: `smoke_hadamard_resid010_400` at `5.1006` subset bpb after int8+zlib roundtrip.
- User-approved hybrid control: `smoke_hadamard_resid010_cosine_lmprobe1_muon_tuned_400` at `2.2588` subset bpb after int8+zlib roundtrip.

## Remember

- Keep the hybrid LM-probe line separate from the strict pure-JEPA line in all comparisons and reporting.
- The current cosine codebook bridge is a major bottleneck: a stop-grad single-layer LM probe decoded the predicted latent much better than the pure JEPA bridge.
- Keep the LM head single-layer for now. The recent gain came from optimizer tuning, not extra decoder depth.
- Optimizer isolation was not the lever: splitting the detached LM head into its own optimizer did not improve the result.
- A naive Muon swap was bad; the gain only appeared after retuning momentum, warmup, per-group LRs, warmdown, and clipping.
- Under the matched `120`-step GA4 control, both a `2`-block causal predictor stack and a naive `5`-layer encoder scale-up underperformed the simpler `4`-layer hybrid baseline while using more artifact budget.
- The same pattern held at `400` steps for the naive `5`-layer encoder scale-up, so the gap is not just an early-training effect.
- MSE-on-MSE is not promising in this family. Even after temperature calibration it trailed the cosine control.
- Preserve run outputs and keep subset-only validation labeled proxy-only until a full validation path is used.

## Open Work

- Preserve the `VAL_MAX_BYTES` smoke path for fast iteration, but keep its scores labeled proxy-only until full validation is run.
- For strict pure JEPA, keep `smoke_hadamard_resid010_400` as the control and focus on stronger targets, bridges, or longer-budget training rather than MSE energy or optimizer splitting.
- For the hybrid line, treat `smoke_hadamard_resid010_cosine_lmprobe1_muon_tuned_400` as the control and start future Muon work from that schedule rather than from the failed naive swap.
- If Muon is revisited again, sweep around the tuned schedule instead of changing the optimizer in isolation.



## 2026-03-27 Contrastive Delta JEPA

- `smoke_ctxdelta_infoncek1_dim192_l2_seq128_cos_vicreg_lmmlp256_w01_5min_g0`
  - Hypothesis: add sampled InfoNCE directly on the predicted delta / teacher delta pair to force sharper token-discriminative deltas.
  - Key settings: `TARGET_SOURCE=contextual_delta`, `CONTRASTIVE_MODE=delta`, `CONTRASTIVE_WEIGHT=0.1`, `CONTRASTIVE_FUTURE_STEPS=1`, `MODEL_DIM=192`, `NUM_LAYERS=2`, `LM_PROBE_INPUT=context_plus_pred`, `LM_PROBE_HIDDEN_DIM=256`.
  - Artifact: `2,732,236` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.784977`, `proxy_val_loss=0.430181`.
  - Decision: discard. Worse than the non-contrastive `2.676675` control.

- `smoke_ctxdelta_infoncek4_dim192_l2_seq128_cos_vicreg_lmmlp256_w01_5min_g1`
  - Hypothesis: contrasting a longer-horizon future delta (`k=4`) may make the target change larger and more discriminative.
  - Key settings: same as above but `CONTRASTIVE_FUTURE_STEPS=4`.
  - Artifact: `2,730,319` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=3.115455`, `proxy_val_loss=0.492110`.
  - Decision: discard. Much worse.

- `smoke_ctxdelta_futurestate_infonce_dim192_l2_seq128_cos_vicreg_lmmlp256_w005_5min_g0`
  - Hypothesis: contrast the reconstructed future state `context + predicted_delta` against the EMA teacher future state, instead of contrasting raw deltas.
  - Key settings: `CONTRASTIVE_MODE=future_state`, `CONTRASTIVE_WEIGHT=0.05`, `CONTRASTIVE_FUTURE_STEPS=1`, `MODEL_DIM=192`, `NUM_LAYERS=2`, `LM_PROBE_INPUT=context_plus_pred`, `LM_PROBE_HIDDEN_DIM=256`.
  - Artifact: `2,884,936` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.469511`, `proxy_val_loss=0.476456`.
  - Decision: keep. New best for the contextual-delta family.

- `smoke_ctxdelta_futurestate_infonce_dim192_l2_seq128_cos_vicreg_lmmlp256_w01_5min_g1`
  - Hypothesis: increase future-state InfoNCE strength from `0.05` to `0.1`.
  - Key settings: same as above but `CONTRASTIVE_WEIGHT=0.1`.
  - Artifact: `2,886,090` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.469609`, `proxy_val_loss=0.478122`.
  - Decision: discard. Essentially tied but slightly worse.

- `smoke_ctxdelta_futurestate_infonce_rawpred3_dim192_l2_seq128_cos_vicreg_lmmlp256_w005_5min_g0`
  - Hypothesis: combine the future-state InfoNCE win with a deeper raw-token predictor encoder.
  - Key settings: same kept branch but `PREDICTOR_MODEL_LAYERS=3`.
  - Artifact: `3,165,646` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.510302`, `proxy_val_loss=0.449117`.
  - Decision: discard. Slightly worse than the kept future-state InfoNCE branch.

- `smoke_ctxdelta_futurestate_infonce_dim256_l2_seq128_cos_vicreg_lmmlp256_w005_5min_g1`
  - Hypothesis: combine the future-state InfoNCE win with a wider latent.
  - Key settings: same kept branch but `MODEL_DIM=256`, `NUM_HEADS=8`, `MLP_HIDDEN_DIM=1024`.
  - Artifact: `4,279,613` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.524372`, `proxy_val_loss=0.469449`.
  - Decision: discard. Slightly worse and slower.


- `smoke_contextualonly_predtarget_dim192_l2_seq128_cos_vicreg_lmmlp256_fsinfo005_5min_g0`
  - Hypothesis: make the predictor directly predict the EMA teacher future embedding (`TARGET_SOURCE=contextual_only`) and decode from that predicted future state.
  - Key settings: `TARGET_SOURCE=contextual_only`, `CONTRASTIVE_MODE=future_state`, `CONTRASTIVE_WEIGHT=0.05`, `LM_PROBE_INPUT=pred_target`, `LM_PROBE_HIDDEN_DIM=256`, `MODEL_DIM=192`, `NUM_LAYERS=2`.
  - Artifact: `2,768,074` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=3.132087`, `proxy_val_loss=0.082675`.
  - Decision: discard. Much worse than the delta-based future-state InfoNCE branch.

- `smoke_contextualonly_ctxplus_dim192_l2_seq128_cos_vicreg_lmmlp256_fsinfo005_5min_g1`
  - Hypothesis: same direct future-embedding target, but let the LM probe see both current context and predicted future state.
  - Key settings: same as above but `LM_PROBE_INPUT=context_plus_pred`.
  - Artifact: `2,750,804` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=3.012011`, `proxy_val_loss=0.083566`.
  - Decision: discard. Slightly better than `pred_target` only, but still far worse than the kept delta-based branch at `2.469511`.


## 2026-03-27 Action, Multi-Step, And Probe-Bridge Follow-Up

- `smoke_actiontr_delta_action_dim64_dim192_l2_seq128_cos_vicreg_fsinfo005_5min_g0`
  - Hypothesis: factor the delta branch into a small action bottleneck and transition model, then decode directly from the action.
  - Key settings: `PREDICTOR_STYLE=action_transition`, `ACTION_DIM=64`, `TARGET_SOURCE=contextual_delta`, `LM_PROBE_INPUT=action`, `CONTRASTIVE_MODE=future_state`, `CONTRASTIVE_WEIGHT=0.05`, `MODEL_DIM=192`, `NUM_LAYERS=2`.
  - Artifact: `3,062,917` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=4.200408`, `proxy_val_loss=0.208690`.
  - Decision: discard. The action bottleneck collapsed token decoding badly.

- `smoke_actiontr_delta_ctxaction_dim64_dim192_l2_seq128_cos_vicreg_fsinfo005_5min_g1`
  - Hypothesis: same action-transition delta branch, but let the LM probe see both state and action.
  - Key settings: same as above but `LM_PROBE_INPUT=context_plus_action`.
  - Artifact: `3,160,188` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=3.665894`, `proxy_val_loss=0.198489`.
  - Decision: discard. Better than action-only, still much worse than the kept delta baseline.

- `smoke_actiontr_future_action_dim64_dim192_l2_seq128_cos_vicreg_fsinfo005_5min_g0`
  - Hypothesis: use the action-transition factorization to predict the full next contextual state instead of the delta target, and decode from the action only.
  - Key settings: `PREDICTOR_STYLE=action_transition`, `ACTION_DIM=64`, `TARGET_SOURCE=contextual_only`, `LM_PROBE_INPUT=action`, `CONTRASTIVE_MODE=future_state`, `CONTRASTIVE_WEIGHT=0.05`, `MODEL_DIM=192`, `NUM_LAYERS=2`.
  - Artifact: `2,995,978` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=4.549748`, `proxy_val_loss=0.026734`.
  - Decision: discard. The future-state loss became extremely easy, but token information was even less recoverable.

- `smoke_actiontr_future_ctxaction_dim64_dim192_l2_seq128_cos_vicreg_fsinfo005_5min_g1`
  - Hypothesis: same future-state action branch, but decode from both state and action.
  - Key settings: same as above but `LM_PROBE_INPUT=context_plus_action`.
  - Artifact: `3,098,340` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=4.135087`, `proxy_val_loss=0.025405`.
  - Decision: discard. State+action helped relative to action-only but was still far worse than the kept contextual-delta branch.

- `smoke_ctxdelta_t2_fsinfo2_dim192_l2_seq128_cos_vicreg_lmmlp256_w005_5min_g0`
  - Hypothesis: the one-step delta may be too small; train JEPA directly on a two-step contextual delta and align the future-state contrastive loss to the same horizon.
  - Key settings: `TARGET_SOURCE=contextual_delta`, `TARGET_FUTURE_STEPS=2`, `CONTRASTIVE_MODE=future_state`, `CONTRASTIVE_FUTURE_STEPS=2`, `LM_PROBE_INPUT=context_plus_pred`, `LM_PROBE_HIDDEN_DIM=256`, `MODEL_DIM=192`, `NUM_LAYERS=2`.
  - Artifact: `2,891,764` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.834102`, `proxy_val_loss=0.574650`.
  - Decision: discard. The larger-horizon JEPA target hurt next-byte decoding.

- `smoke_ctxdelta_t4_fsinfo4_dim192_l2_seq128_cos_vicreg_lmmlp256_w005_5min_g1`
  - Hypothesis: push the delta/future-state target farther to four steps to make the state change even more discriminative.
  - Key settings: same as above but `TARGET_FUTURE_STEPS=4`, `CONTRASTIVE_FUTURE_STEPS=4`.
  - Artifact: `2,893,796` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=3.276485`, `proxy_val_loss=0.603690`.
  - Decision: discard. Longer-horizon delta targets moved farther away from the next-byte probe.

- `smoke_ema_static_ctxplus_lmmlp256_seq256_cosine_lmprobe05_5min_g0`
  - Hypothesis: the best static EMA JEPA backbone may already be strong enough, and the bottleneck may be the weak linear probe; add a single-hidden-layer MLP and expose both context and predicted latent.
  - Key settings: `TARGET_SOURCE=codebook`, `TARGET_CODEBOOK=ema`, `TRAIN_SEQ_LEN=256`, `MODEL_DIM=256`, `NUM_LAYERS=4`, `PREDICTOR_HIDDEN_DIM=1024`, `LM_PROBE_INPUT=context_plus_pred`, `LM_PROBE_HIDDEN_DIM=256`, `LM_PROBE_WEIGHT=0.5`.
  - Artifact: `3,082,750` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=1.970319`, `proxy_val_loss=0.556663`.
  - Decision: keep as an improved decode-bridge variant, but it is not the best MLP-head setting.

- `smoke_ema_static_pred_lmmlp256_seq256_cosine_lmprobe05_5min_g1`
  - Hypothesis: keep the same static EMA JEPA backbone, but replace the detached linear probe with a shallow one-hidden-layer MLP on the predicted latent alone.
  - Key settings: same as above but `LM_PROBE_INPUT=pred_target`.
  - Artifact: `3,141,715` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=1.954263`, `proxy_val_loss=0.556174`.
  - Decision: keep. New best 5-minute branch and the first major gain from improving the decode bridge instead of the JEPA target.

- `smoke_ema_static_pred_lmmlp512_seq256_cosine_lmprobe05_5min_g0`
  - Hypothesis: a slightly wider single-hidden-layer MLP probe may further improve the decode bridge.
  - Key settings: same kept branch but `LM_PROBE_HIDDEN_DIM=512`.
  - Artifact: `3,116,154` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=1.989003`, `proxy_val_loss=0.553094`.
  - Decision: discard. Worse than the `256`-hidden probe.

- `smoke_ema_static_pred_lmmlp768_seq256_cosine_lmprobe05_5min_g1`
  - Hypothesis: continue widening the shallow MLP probe to see whether decoding is still the dominant bottleneck.
  - Key settings: same kept branch but `LM_PROBE_HIDDEN_DIM=768`.
  - Artifact: `3,208,372` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=1.958073`, `proxy_val_loss=0.554889`.
  - Decision: discard. Essentially tied but still worse than the `256`-hidden probe while using more bytes.

- `smoke_ema_static_pred_linear_seq256_cosine_lmprobe05_10min_g0`
  - Hypothesis: run the old linear-head static EMA control for the full 10-minute budget to compare fairly against the new MLP-head branch.
  - Key settings: static EMA control, `LM_PROBE_INPUT=pred_target`, `LM_PROBE_HIDDEN_DIM=0`, `MAX_WALLCLOCK_SECONDS=600`, `WARMDOWN_ITERS=6000`.
  - Artifact: `3,176,252` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.054000`, `proxy_val_loss=0.564052`.
  - Decision: keep as the 10-minute control.

- `smoke_ema_static_pred_lmmlp256_seq256_cosine_lmprobe05_10min_g1`
  - Hypothesis: the shallow MLP probe improvement should persist or grow over the full 10-minute budget.
  - Key settings: same backbone as the 10-minute control but `LM_PROBE_HIDDEN_DIM=256`.
  - Artifact: `3,359,711` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=1.844254`, `proxy_val_loss=0.559508`.
  - Decision: keep. New best result so far. The static EMA JEPA backbone appears substantially stronger than the original linear decode bridge suggested.


- `smoke_ema_static_rawpred2_pred_lmmlp256_seq256_cosine_lmprobe05_5min_g0`
  - Hypothesis: with the stronger MLP probe in place, a separate raw-token predictor encoder may improve the JEPA latent itself.
  - Key settings: static EMA branch, `PREDICTOR_INPUT=tokens`, `PREDICTOR_MODEL_LAYERS=2`, `LM_PROBE_HIDDEN_DIM=256`.
  - Artifact: `4,236,584` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.060263`, `proxy_val_loss=0.707267`.
  - Decision: discard. Worse than the kept static MLP-probe branch while using much more budget.

- `smoke_ema_static_predstack2_pred_lmmlp256_seq256_cosine_lmprobe05_5min_g1`
  - Hypothesis: add a lightweight causal predictor stack on top of the context latent now that decoding is no longer the main bottleneck.
  - Key settings: static EMA branch, `PREDICTOR_NUM_LAYERS=2`, `PREDICTOR_NUM_HEADS=4`, `LM_PROBE_HIDDEN_DIM=256`.
  - Artifact: `4,143,457` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.109115`, `proxy_val_loss=0.547828`.
  - Decision: discard. Clear regression.

- `smoke_ema_static_dim192_pred768_lmmlp256_seq256_cosine_lmprobe05_5min_g0`
  - Hypothesis: under a fixed wallclock, a smaller static EMA backbone might beat the 256d model by taking more updates once the probe is fixed.
  - Key settings: `MODEL_DIM=192`, `NUM_HEADS=6`, `MLP_HIDDEN_DIM=768`, `PREDICTOR_HIDDEN_DIM=768`, `LM_PROBE_HIDDEN_DIM=256`.
  - Artifact: `2,166,385` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=1.968194`, `proxy_val_loss=0.561580`.
  - Decision: discard for 5-minute best-of, but keep in mind as a strong small-model tradeoff.

- `smoke_ema_static_dim128_pred512_lmmlp256_seq256_cosine_lmprobe05_5min_g1`
  - Hypothesis: push the size/steps tradeoff harder with a very small static EMA JEPA backbone and the improved MLP probe.
  - Key settings: `MODEL_DIM=128`, `NUM_HEADS=4`, `MLP_HIDDEN_DIM=512`, `PREDICTOR_HIDDEN_DIM=512`, `LM_PROBE_HIDDEN_DIM=256`.
  - Artifact: `2,158,223` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=1.952371`, `proxy_val_loss=0.596718`.
  - Decision: keep as an interesting small-model budget tradeoff. It is roughly tied with the 256d 5-minute best after int8 roundtrip, but the raw proxy is slightly worse.

- `smoke_ema_static_dim192_pred768_lmmlp256_seq256_cosine_lmprobe05_10min_g1`
  - Hypothesis: the 192d static EMA branch may scale well under the full 10-minute budget because it gets more updates than the 256d model.
  - Key settings: same as the kept 192d 5-minute branch but `MAX_WALLCLOCK_SECONDS=600`, `WARMDOWN_ITERS=6000`.
  - Artifact: `2,340,453` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=1.861303`, `proxy_val_loss=0.566039`.
  - Decision: keep as a very strong small-model baseline, but it still trails the 256d 10-minute winner.

- `smoke_ema_static_dim128_pred512_lmmlp256_seq256_cosine_lmprobe05_10min_g0`
  - Hypothesis: the tiny 128d branch may win outright under the full 10-minute budget by taking far more updates.
  - Key settings: same as the kept 128d 5-minute branch but `MAX_WALLCLOCK_SECONDS=600`, `WARMDOWN_ITERS=6000`.
  - Artifact: `2,161,869` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=1.919642`, `proxy_val_loss=0.594488`.
  - Decision: discard. Extra steps were not enough to overcome the smaller latent.


## 2026-03-28 Rotation-Scale Delta JEPA

- `smoke_rotscale_diffloss_diffprobe_cos_5min_g0`
  - Hypothesis: replace the additive contextual delta with a rotation-plus-log-scale diff code, train JEPA directly on that diff with cosine loss, and decode from the diff alone.
  - Key settings: `TARGET_SOURCE=rotation_scale_delta`, `ROTATION_DELTA_LOSS_TARGET=diff`, `JEPA_LOSS=cosine`, `LM_PROBE_INPUT=pred_target`, `LM_PROBE_HIDDEN_DIM=256`, `MODEL_DIM=192`, `NUM_LAYERS=2`, raw-token predictor encoder.
  - Artifact: `2,770,047` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.726460`, `proxy_val_loss=0.412059`.
  - Decision: keep as the baseline for this family. The geometric diff is decodable, but not enough on its own to beat older contextual-delta runs.

- `smoke_rotscale_diffloss_ctxdiffprobe_cos_5min_g1`
  - Hypothesis: same geometric diff target, but let the detached LM probe see both context and predicted diff.
  - Key settings: same as above but `LM_PROBE_INPUT=context_plus_pred`.
  - Artifact: `2,749,927` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.693380`, `proxy_val_loss=0.413404`.
  - Decision: keep. Best result in the rotation-scale family so far.

- `smoke_rotscale_nextloss_diffprobe_cos_5min_g0`
  - Hypothesis: still predict the rotation-scale diff, but apply the JEPA loss on the reconstructed next state instead of the diff code; decode from diff only.
  - Key settings: `ROTATION_DELTA_LOSS_TARGET=next_state`, `JEPA_LOSS=cosine`, `LM_PROBE_INPUT=pred_target`, rest matched to the baseline.
  - Artifact: `2,656,318` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=4.686762`, `proxy_val_loss=0.406524`.
  - Decision: discard. Near-zero-style next-state matching again failed to preserve token information.

- `smoke_rotscale_nextloss_ctxdiffprobe_cos_5min_g1`
  - Hypothesis: same reconstructed-next-state objective, but decode from context plus diff.
  - Key settings: same as above but `LM_PROBE_INPUT=context_plus_pred`.
  - Artifact: `2,625,177` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=4.688177`, `proxy_val_loss=0.386948`.
  - Decision: discard. Same collapse pattern as `diff`-only.

- `smoke_rotscale_diffloss_diffprobe_mse_5min_g0`
  - Hypothesis: the rotation-scale diff has meaningful magnitude, so MSE may fit it better than cosine.
  - Key settings: `ROTATION_DELTA_LOSS_TARGET=diff`, `JEPA_LOSS=mse`, `LM_PROBE_INPUT=pred_target`, rest matched to the cosine baseline.
  - Artifact: `2,771,499` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=3.053223`, `proxy_val_loss=0.250152`.
  - Decision: discard. MSE hurt decode quality despite lowering proxy loss.

- `smoke_rotscale_diffloss_ctxdiffprobe_mse_5min_g1`
  - Hypothesis: same MSE diff objective, but give the probe both context and diff.
  - Key settings: same as above but `LM_PROBE_INPUT=context_plus_pred`.
  - Artifact: `2,760,369` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.952642`, `proxy_val_loss=0.250198`.
  - Decision: discard. Better than diff-only MSE, still worse than cosine.

- `smoke_rotscale_diffloss_ctxdiffprobe_cos_fsnce005_5min_g1`
  - Hypothesis: add SimCLR / InfoNCE on the reconstructed next state while keeping the cosine diff objective.
  - Key settings: cosine diff objective, `LM_PROBE_INPUT=context_plus_pred`, `CONTRASTIVE_MODE=future_state`, `CONTRASTIVE_WEIGHT=0.05`.
  - Artifact: `2,771,438` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=4.373265`, `proxy_val_loss=0.350711`.
  - Decision: discard. Future-state contrastive badly hurt this representation.

- `smoke_rotscale_diffloss_ctxdiffprobe_cos_deltance005_5min_g0`
  - Hypothesis: instead of contrasting the reconstructed next state, contrast the rotation-scale diff code directly.
  - Key settings: cosine diff objective, `LM_PROBE_INPUT=context_plus_pred`, `CONTRASTIVE_MODE=delta`, `CONTRASTIVE_WEIGHT=0.05`.
  - Artifact: `2,719,635` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.832154`, `proxy_val_loss=0.430412`.
  - Decision: discard. Less harmful than future-state contrastive, but still worse than no contrastive term.


- `smoke_ctxdelta_futurestate_infonce_dim192_l2_seq128_cos_sigreg_lmmlp256_w005_5min_g0`
  - Hypothesis: the kept contextual-delta future-state InfoNCE branch may prefer sigma-only regularization (`sigreg`) over the earlier VICReg-style `var+cov` regularizer.
  - Key settings: matched to the kept branch `smoke_ctxdelta_futurestate_infonce_dim192_l2_seq128_cos_vicreg_lmmlp256_w005_5min_g0` but with `VAR_REG_WEIGHT=1.0` and `COV_REG_WEIGHT=0.0`.
  - Artifact: `2,903,251` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.476897`, `proxy_val_loss=0.481402`.
  - Decision: discard. Slightly better raw validation before roundtrip (`2.441599` vs `2.469511`), but slightly worse after the int8+zlib roundtrip than the kept VICReg branch (`2.476897` vs `2.469511`).


## 2026-03-28 Sphere-Delta JEPA

- `smoke_spheredelta_next_ctxplus_cos_5min_g0`
  - Hypothesis: use unit-normalized teacher states, predict an unnormalized tangent-space residual, reconstruct the unit next state, and apply cosine JEPA loss on that reconstructed next state.
  - Key settings: `TARGET_SOURCE=sphere_delta`, `ROTATION_DELTA_LOSS_TARGET=next_state`, `JEPA_LOSS=cosine`, `LM_PROBE_INPUT=context_plus_pred`, `LM_PROBE_HIDDEN_DIM=256`, `MODEL_DIM=192`, `NUM_LAYERS=2`, raw-token predictor encoder, `VAR_REG_WEIGHT=1.0`, `COV_REG_WEIGHT=0.01`.
  - Artifact: `2,668,594` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=4.848183`, `proxy_val_loss=1.072199`.
  - Decision: discard. Reconstructed-next-state supervision again made the latent too easy to match and too weak for token identity.

- `smoke_spheredelta_diff_ctxplus_cos_5min_g1`
  - Hypothesis: keep the same unit-sphere tangent target, but train JEPA directly on the unnormalized tangent diff instead of the reconstructed next state.
  - Key settings: `TARGET_SOURCE=sphere_delta`, `ROTATION_DELTA_LOSS_TARGET=diff`, `JEPA_LOSS=cosine`, `LM_PROBE_INPUT=context_plus_pred`, `LM_PROBE_HIDDEN_DIM=256`, same `192d/2-layer` raw-token predictor stack.
  - Artifact: `2,764,126` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.675118`, `proxy_val_loss=0.409297`.
  - Decision: keep as the best sphere-delta result. Viable, but still behind the older contextual-delta future-state InfoNCE branch.

- `smoke_spheredelta_diff_diffprobe_cos_5min_g0`
  - Hypothesis: the tangent diff itself may already be token-like enough, so the detached LM probe may not need the current context concatenated.
  - Key settings: same as the kept sphere-delta diff run but `LM_PROBE_INPUT=pred_target`.
  - Artifact: `2,777,170` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.737426`, `proxy_val_loss=0.410302`.
  - Decision: discard. Context still helps decode the sphere-delta representation.

- `smoke_spheredelta_diff_ctxplus_cos_fsnce005_5min_g1`
  - Hypothesis: a small future-state InfoNCE term may sharpen the reconstructed target while keeping the tangent-diff JEPA objective.
  - Key settings: same as the kept sphere-delta diff run but `CONTRASTIVE_MODE=future_state`, `CONTRASTIVE_WEIGHT=0.05`.
  - Artifact: `2,821,020` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=4.671547`, `proxy_val_loss=0.777782`.
  - Decision: discard. Future-state contrastive badly destabilized this branch, similar to the earlier rotation-scale failure mode.


## 2026-03-28 Additive-Delta JEPA

- `smoke_additivedelta_none_mse_ctxplus_5min_g0`
  - Hypothesis: predict the raw additive teacher delta `z_{t+1} - z_t` with no normalization anywhere in target construction, and let the detached probe read `[context, diff]`.
  - Key settings: `TARGET_SOURCE=additive_delta`, `ADDITIVE_DELTA_NORMALIZATION=none`, `ROTATION_DELTA_LOSS_TARGET=diff`, `JEPA_LOSS=mse`, `LM_PROBE_INPUT=context_plus_pred`, `LM_PROBE_HIDDEN_DIM=256`, `MODEL_DIM=192`, `NUM_LAYERS=2`, raw-token predictor encoder.
  - Artifact: `2,789,897` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.676848`, `proxy_val_loss=0.309780`.
  - Decision: keep as the best additive-delta result. Raw, unnormalized additive delta is the right version of this family, but it still trails the kept contextual-delta branch and is essentially tied with sphere-delta.

- `smoke_additivedelta_states_mse_ctxplus_5min_g1`
  - Hypothesis: normalize the teacher context and next-state vectors first, then predict their additive diff with MSE.
  - Key settings: same as above but `ADDITIVE_DELTA_NORMALIZATION=states`.
  - Artifact: `2,782,286` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=3.020410`, `proxy_val_loss=0.251294`.
  - Decision: discard. Normalizing teacher states before forming the additive delta hurts badly.

- `smoke_additivedelta_diffnorm_mse_ctxplus_5min_g0`
  - Hypothesis: keep raw teacher states, but normalize only the target diff vector before MSE.
  - Key settings: same additive-delta MSE setup but `ADDITIVE_DELTA_NORMALIZATION=diff`.
  - Artifact: `2,780,464` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.817287`, `proxy_val_loss=0.251757`.
  - Decision: discard. Normalizing only the diff removes some useful magnitude information.

- `smoke_additivedelta_statesdiff_mse_ctxplus_5min_g1`
  - Hypothesis: normalize both teacher states and the resulting diff.
  - Key settings: same additive-delta MSE setup but `ADDITIVE_DELTA_NORMALIZATION=states_and_diff`.
  - Artifact: `2,789,168` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.779814`, `proxy_val_loss=0.253559`.
  - Decision: discard. Slightly better than `diff`-only normalization, still worse than leaving the teacher delta raw.

- `smoke_additivedelta_none_cos_ctxplus_5min_g0`
  - Hypothesis: keep the best additive-delta normalization (`none`) but swap the JEPA loss from MSE to cosine.
  - Key settings: same as the kept additive-delta run but `JEPA_LOSS=cosine`.
  - Artifact: `2,760,676` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.707852`, `proxy_val_loss=0.401916`.
  - Decision: discard. For raw additive delta, MSE is slightly better than cosine after roundtrip.

- `smoke_additivedelta_diffnorm_cos_ctxplus_5min_g1`
  - Hypothesis: if cosine is used anyway, explicitly normalizing the target diff may align the objective better.
  - Key settings: `ADDITIVE_DELTA_NORMALIZATION=diff`, `JEPA_LOSS=cosine`, rest matched to the kept additive run.
  - Artifact: `2,772,554` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.680375`, `proxy_val_loss=0.398832`.
  - Decision: keep as a near-tie diagnostic. Cosine largely swallows the explicit diff normalization choice, but it still does not beat the raw-MSE additive delta.

## 2026-03-30 Raw Contextual Delta

- `smoke_rawctxdelta_mse_fsnce005_5min_g1`
  - Hypothesis: keep the strongest contextual-delta recipe, but stop normalizing the orthogonal teacher residual so the predictor can learn the magnitude of the new direction as well as its direction.
  - Key settings: `TARGET_SOURCE=contextual_delta`, `CONTEXTUAL_DELTA_NORMALIZATION=raw`, `JEPA_LOSS=mse`, `CONTRASTIVE_MODE=future_state`, `CONTRASTIVE_WEIGHT=0.05`, `VAR_REG_WEIGHT=1.0`, `COV_REG_WEIGHT=0.01`, `LM_PROBE_INPUT=context_plus_pred`, `LM_PROBE_HIDDEN_DIM=256`, `MODEL_DIM=192`, `NUM_LAYERS=2`, raw-token predictor encoder.
  - Artifact: `2,907,032` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.399791`, `proxy_val_loss=0.626322`.
  - Decision: keep. This is the new best contextual target family result so far, beating the normalized contextual-delta best (`2.469511`) by about `0.07` bpb after roundtrip.

- `smoke_rawctxdelta_cos_fsnce005_5min_g0`
  - Hypothesis: the same raw contextual delta may still prefer cosine once magnitude is restored, with future-state InfoNCE kept on.
  - Key settings: same as the kept raw-contextual run but `JEPA_LOSS=cosine`.
  - Artifact: `2,896,318` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.404920`, `proxy_val_loss=0.467873`.
  - Decision: discard. Very close, but still slightly worse than raw-contextual `mse` after roundtrip.

- `smoke_rawctxdelta_mse_nocontrast_5min_g0`
  - Hypothesis: if restored magnitude is the real missing signal, the raw contextual delta might work without any future-state contrastive term.
  - Key settings: same as the kept raw-contextual run but `CONTRASTIVE_WEIGHT=0.0`.
  - Artifact: `2,780,884` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.677045`, `proxy_val_loss=0.308449`.
  - Decision: discard. Removing future-state InfoNCE wipes out most of the gain, so magnitude alone is not enough.

- `smoke_rawctxdelta_cos_nocontrast_5min_g1`
  - Hypothesis: cosine might be a better match than MSE for raw contextual delta once contrastive pressure is removed.
  - Key settings: same no-contrastive raw-contextual setup but `JEPA_LOSS=cosine`.
  - Artifact: `2,772,256` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.667598`, `proxy_val_loss=0.409528`.
  - Decision: discard. Slightly better than raw-MSE without contrastive, still far worse than keeping the future-state contrastive term.

- `smoke_rawctxdelta_mse_fsnce005_sigreg_5min_g0`
  - Hypothesis: the raw contextual branch may only need variance pressure, making the small covariance term unnecessary.
  - Key settings: same as the kept raw-contextual run but `COV_REG_WEIGHT=0.0`.
  - Artifact: `2,902,883` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.404613`, `proxy_val_loss=0.622201`.
  - Decision: discard. `sigreg` is effectively a tie in raw proxy loss but slightly worse after int8+zlib roundtrip, so keep the small covariance penalty.

## 2026-03-30 Raw Contextual Delta With Parallel Shift Head

- `smoke_rawctxdelta_parallel01_ctxplus_5min_g0`
  - Hypothesis: add a tiny scalar head for the contextual-delta parallel shift, use it to reconstruct the future state for InfoNCE, but keep the LM probe input unchanged as `[context, pred_delta]`.
  - Key settings: `TARGET_SOURCE=contextual_delta`, `CONTEXTUAL_DELTA_NORMALIZATION=raw`, `CONTEXTUAL_DELTA_PARALLEL_WEIGHT=0.1`, `JEPA_LOSS=mse`, `CONTRASTIVE_MODE=future_state`, `CONTRASTIVE_WEIGHT=0.05`, `LM_PROBE_INPUT=context_plus_pred`, `LM_PROBE_HIDDEN_DIM=256`, `MODEL_DIM=192`, `NUM_LAYERS=2`, raw-token predictor encoder.
  - Artifact: `2,963,649` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.566098`, `proxy_val_loss=0.756442`.
  - Decision: discard. The extra scalar supervision is too strong at this weight and degrades the branch substantially.

- `smoke_rawctxdelta_parallel01_ctxplusscalar_5min_g1`
  - Hypothesis: if the scalar head is useful, letting the detached probe see it should help decode the next token.
  - Key settings: same as above but `LM_PROBE_INPUT=context_plus_pred_parallel`.
  - Artifact: `2,956,411` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.584012`, `proxy_val_loss=0.767162`.
  - Decision: discard. Exposing the scalar to the probe does not rescue the branch.

- `smoke_rawctxdelta_parallel001_ctxplus_5min_g0`
  - Hypothesis: the scalar head may only need a weak auxiliary weight because the parallel-shift target lives on a smaller scale than the raw orthogonal residual.
  - Key settings: same as the first parallel-head run but `CONTEXTUAL_DELTA_PARALLEL_WEIGHT=0.01`.
  - Artifact: `2,952,378` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.546429`, `proxy_val_loss=0.716885`.
  - Decision: discard. Lowering the weight helps relative to `0.1`, but the branch still trails the kept raw-contextual control.

- `smoke_rawctxdelta_parallel001_ctxplusscalar_5min_g1`
  - Hypothesis: combine the lighter scalar-loss weight with probe access to the scalar.
  - Key settings: same as above but `LM_PROBE_INPUT=context_plus_pred_parallel`.
  - Artifact: `2,952,833` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.523575`, `proxy_val_loss=0.725517`.
  - Decision: discard. This is the best parallel-head variant, but it still underperforms the kept raw-contextual control at `2.399791`.

## 2026-03-30 Contextual Delta Direction Plus Log-Norm

- `smoke_ctxdelta_dirlog_scaledprobe_5min_g0`
  - Hypothesis: factor the orthogonal residual into unit direction plus log-norm, train the direction with cosine loss, predict the log-norm with a small scalar head, and let the probe decode from `[context, scaled_delta]`.
  - Key settings: `TARGET_SOURCE=contextual_delta`, `CONTEXTUAL_DELTA_NORMALIZATION=normalized`, `CONTEXTUAL_DELTA_MAGNITUDE_WEIGHT=0.01`, `JEPA_LOSS=cosine`, `CONTRASTIVE_MODE=future_state`, `CONTRASTIVE_WEIGHT=0.05`, `LM_PROBE_INPUT=context_plus_scaled_pred`, `LM_PROBE_HIDDEN_DIM=256`, `MODEL_DIM=192`, `NUM_LAYERS=2`, raw-token predictor encoder.
  - Artifact: `2,898,266` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.565182`, `proxy_val_loss=0.387663`.
  - Decision: discard. Separating direction and magnitude this way is materially worse than the simpler raw contextual-delta target.

- `smoke_ctxdelta_dirlog_magprobe_5min_g1`
  - Hypothesis: if the factorized branch works, exposing the predicted log-norm directly to the detached probe should help decode the next token.
  - Key settings: same as above but `LM_PROBE_INPUT=context_plus_pred_magnitude`.
  - Artifact: `2,892,043` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.580809`, `proxy_val_loss=0.395364`.
  - Decision: discard. Giving the probe the explicit log-norm does not rescue the branch.

## 2026-03-30 Context-Latent Predictor Scaling

- `smoke_rawctxdelta_ctxlatent_dim256_10min_g1`
  - Hypothesis: the raw contextual-delta branch is predictor-limited, so widening the encoder state while keeping the simple context-latent predictor may improve token alignment.
  - Key settings: `TARGET_SOURCE=contextual_delta`, `CONTEXTUAL_DELTA_NORMALIZATION=raw`, `TARGET_ENCODER_CAUSAL=1`, `PREDICTOR_INPUT=context_latent`, `MODEL_DIM=256`, `NUM_LAYERS=2`, `PREDICTOR_HIDDEN_DIM=1024`, `JEPA_LOSS=mse`, `CONTRASTIVE_MODE=future_state`, `CONTRASTIVE_WEIGHT=0.05`, `LM_PROBE_INPUT=context_plus_pred`, `LM_PROBE_HIDDEN_DIM=256`, `LM_PROBE_WEIGHT=0.5`, `VAR_REG_WEIGHT=1.0`, `COV_REG_WEIGHT=0.01`, `10` minute cap.
  - Artifact: `3,855,773` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.175443`, `proxy_val_loss=0.285774`.
  - Decision: keep. This became the strongest contextual branch before predictor-side sequence modeling.

- `smoke_rawctxdelta_ctxlatent_dim256_predhid1536_10min_g0`
  - Hypothesis: the `256d` context-latent branch may want a wider MLP predictor head before changing the predictor structure.
  - Key settings: same as above but `PREDICTOR_HIDDEN_DIM=1536`.
  - Artifact: `3,909,618` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.192491`, `proxy_val_loss=0.293850`.
  - Decision: discard. A wider plain MLP predictor regressed after roundtrip.

- `smoke_rawctxdelta_ctxlatent_dim320_10min_g1`
  - Hypothesis: if the contextual branch still lacks representation capacity, a wider encoder state should help more than changing the predictor.
  - Key settings: `MODEL_DIM=320`, `NUM_LAYERS=2`, `MLP_HIDDEN_DIM=1280`, `PREDICTOR_HIDDEN_DIM=1024`, otherwise matched to the kept `256d` branch.
  - Artifact: `4,675,908` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.176373`, `proxy_val_loss=0.308277`.
  - Decision: discard. Nearly tied the `256d` branch while costing more bytes and wallclock; width alone is not the lever.

## 2026-03-30 Teacher-Geometry Contextual Delta

- `smoke_rawctxdelta_ctxlatent_dim256_teacheranchor_10min_g0`
  - Hypothesis: the future-state InfoNCE term may be misaligned because it reconstructs from the online context anchor instead of the teacher anchor, so switching the contrastive anchor to the teacher state might sharpen the target without changing the main JEPA loss.
  - Key settings: matched to `smoke_rawctxdelta_ctxlatent_dim256_10min_g1` but `CONTEXTUAL_DELTA_CONTRASTIVE_TEACHER_ANCHOR=1`.
  - Artifact: `3,728,256` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.476694`, `proxy_val_loss=0.147902`.
  - Decision: discard. This made the latent objective much easier while hurting token prediction badly.

- `smoke_rawctxdelta_ctxlatent_dim256_teacheranchor_orth002_10min_g1`
  - Hypothesis: add a light penalty on the predicted component parallel to the teacher state to force the predictor into the same tangent space as the raw contextual-delta target.
  - Key settings: same as above plus `CONTEXTUAL_DELTA_TEACHER_ORTH_WEIGHT=0.02`.
  - Artifact: `3,786,212` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.522097`, `proxy_val_loss=0.149877`.
  - Decision: discard. Stronger geometry matching further improved proxy loss while degrading bpb even more.

## 2026-03-30 Sequence-Aware Predictor On Context Latents

- `smoke_rawctxdelta_ctxlatent_predstack1_dim256_10min_g0`
  - Hypothesis: the context-to-delta map may need one causal sequence-modeling block in the predictor, not just a per-position MLP.
  - Key settings: matched to the kept `256d` branch but `PREDICTOR_NUM_LAYERS=1`, `PREDICTOR_NUM_HEADS=8`.
  - Artifact: `4,205,911` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.153404`, `proxy_val_loss=0.320739`.
  - Decision: keep. Adding a single predictor block helped more than encoder widening.

- `smoke_rawctxdelta_ctxlatent_predstack1_predhid1536_dim256_10min_g1`
  - Hypothesis: once the predictor gets one causal block, a wider MLP head may unlock more of the token information already present in the contextual delta target.
  - Key settings: same as above but `PREDICTOR_HIDDEN_DIM=1536`.
  - Artifact: `4,625,666` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.117175`, `proxy_val_loss=0.311524`.
  - Decision: keep. This is the current best contextual JEPA run so far.

- `smoke_rawctxdelta_ctxlatent_predstack1_predhid2048_dim256_10min_g0`
  - Hypothesis: the `1`-block predictor may still benefit from more width before the branch saturates.
  - Key settings: same as the kept `1`-block branch but `PREDICTOR_HIDDEN_DIM=2048`.
  - Artifact: `4,784,647` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.131578`, `proxy_val_loss=0.307599`.
  - Decision: discard. Slightly worse after roundtrip; `1536` looks near the sweet spot.

- `smoke_rawctxdelta_ctxlatent_predstack1_predhid1536_dim320_10min_g1`
  - Hypothesis: combine the winning `1`-block predictor with a wider encoder state to see if the newly useful predictor can exploit more context capacity.
  - Key settings: `MODEL_DIM=320`, `NUM_LAYERS=2`, `MLP_HIDDEN_DIM=1280`, `PREDICTOR_NUM_LAYERS=1`, `PREDICTOR_HIDDEN_DIM=1536`, otherwise matched to the kept branch.
  - Artifact: `5,527,246` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.133358`, `proxy_val_loss=0.315232`.
  - Decision: discard. The predictor-side gain does not transfer to a wider encoder under the same wallclock cap.

## 2026-03-30 Shared-Encoder SigReg Pilots

- `pilot_rawctxdelta_sharedsg_sigreg_nocontrast_5min_g0`
  - Hypothesis: remove EMA and contrastive pressure, share the online encoder for target generation, keep stop-grad on the target states, and rely on sigma regularization alone to keep the latent geometry useful.
  - Key settings: `TARGET_SOURCE=contextual_delta`, `CONTEXTUAL_DELTA_NORMALIZATION=raw`, `TARGET_ENCODER_MODE=shared`, `TARGET_STOP_GRAD=1`, `CONTRASTIVE_WEIGHT=0.0`, `VAR_REG_WEIGHT=1.0`, `COV_REG_WEIGHT=0.0`, `MODEL_DIM=256`, `NUM_LAYERS=2`, `PREDICTOR_INPUT=context_latent`, `PREDICTOR_NUM_LAYERS=1`, `PREDICTOR_HIDDEN_DIM=1536`, `5` minute cap, reduced `TRAIN_BATCH_BYTES=393216` for a safe pilot.
  - Artifact: `2,793,367` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=3.832371`, `proxy_val_loss=0.072588`.
  - Decision: discard. Even the stop-grad shared-target version collapses to an easy latent objective with very poor token prediction.

- `pilot_rawctxdelta_sharednosg_sigreg_nocontrast_5min_g1`
  - Hypothesis: match recent `LeJEPA`-style symmetry more directly by removing both EMA and stop-grad, keeping only `sigreg` with the shared encoder target path.
  - Key settings: same as the previous pilot but `TARGET_STOP_GRAD=0`.
  - Artifact: `2,521,302` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=4.676206`, `proxy_val_loss=0.003917`.
  - Decision: discard hard. The latent objective becomes almost trivial while token prediction collapses even further, so this symmetry is not viable as a direct swap in the current byte-level setup.

- `pilot_ctxdelta_norm_sharednosg_mse_sigreg_nocontrast_5min_g0`
  - Hypothesis: if the no-stop-grad shared-target variant is abusing delta magnitude, forcing the target onto the unit sphere should make the objective less degenerate and recover some token signal.
  - Key settings: same as the previous no-stop-grad pilot but `CONTEXTUAL_DELTA_NORMALIZATION=normalized` and `JEPA_LOSS=mse`.
  - Artifact: `2,579,968` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=4.604065`, `proxy_val_loss=0.251910`.
  - Decision: discard. Normalizing the delta helps only marginally on bpb and still leaves the no-stop-grad shared-target family far behind viable contextual runs.

- `pilot_ctxdelta_norm_sharednosg_cos_sigreg_nocontrast_5min_g1`
  - Hypothesis: combine the normalized shared-target delta with cosine loss so the objective matches the unit-norm target geometry directly.
  - Key settings: same as the previous normalized pilot but `JEPA_LOSS=cosine`.
  - Artifact: `2,578,430` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=4.658614`, `proxy_val_loss=0.001488`.
  - Decision: discard hard. Cosine on the normalized no-stop-grad target makes the latent objective nearly trivial again without restoring token discrimination.

- `pilot_rawctxdelta_sharednosg_sigreg_targetsig_5min_g0`
  - Hypothesis: the no-stop-grad shared-target branch may need sigma regularization on the teacher delta targets as well as the predicted deltas, so the whole shared latent geometry stays spread out.
  - Key settings: same as `pilot_rawctxdelta_sharednosg_sigreg_nocontrast_5min_g1` but with `TARGET_VAR_REG_WEIGHT=1.0`, `TARGET_COV_REG_WEIGHT=0.0`.
  - Artifact: `2,515,406` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=4.680427`, `proxy_val_loss=0.005616`.
  - Decision: discard hard. Target-side sigreg on raw deltas did not help; it slightly regressed bpb while keeping the latent objective nearly trivial.

- `pilot_ctxdelta_norm_sharednosg_mse_sigreg_targetsig_5min_g0`
  - Hypothesis: combine target-side sigreg with normalized contextual deltas so the no-stop-grad target path cannot cheat through either norm collapse or packed target geometry.
  - Key settings: same as `pilot_ctxdelta_norm_sharednosg_mse_sigreg_nocontrast_5min_g0` but with `TARGET_VAR_REG_WEIGHT=1.0`, `TARGET_COV_REG_WEIGHT=0.0`.
  - Artifact: `2,556,153` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=4.622339`, `proxy_val_loss=0.253231`.
  - Decision: discard. This is also slightly worse than the matched normalized no-target-sigreg baseline, so target-side sigreg does not rescue the no-stop-grad shared family.

## 2026-03-31 No-InfoNCE Future-Step Sweep

- `pilot_rawctxdelta_ctxlatent_predstack1_predhid1536_dim256_nocontrast_h1_anchor_5min_g0`
  - Hypothesis: before replacing InfoNCE, measure how strong the current best contextual architecture is with `EMA + stop-grad + sigreg/vicreg` alone and no contrastive term.
  - Key settings: matched to the kept `256d` contextual branch but `CONTRASTIVE_WEIGHT=0.0`, `TARGET_FUTURE_STEPS=1`, `CONTEXTUAL_DELTA_REFERENCE=anchor`, `5` minute cap.
  - Artifact: `4,039,113` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.231334`, `proxy_val_loss=0.370658`.
  - Decision: keep as the no-InfoNCE baseline. This is much better than the earlier raw-token predictor no-contrastive branch and shows the stronger context-latent predictor carries most of the performance.

- `pilot_rawctxdelta_ctxlatent_predstack1_predhid1536_dim256_nocontrast_h2_anchor_5min_g0`
  - Hypothesis: replace the next-step target with an anchor-relative `h=2` delta so the model predicts a more substantial future change without using contrastive pressure.
  - Key settings: same as the no-InfoNCE baseline but `TARGET_FUTURE_STEPS=2`.
  - Artifact: `4,009,596` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.625305`, `proxy_val_loss=0.295921`.
  - Decision: discard. The latent objective gets easier, but the model loses next-byte sharpness.

- `pilot_rawctxdelta_ctxlatent_predstack1_predhid1536_dim256_nocontrast_h4_anchor_5min_g0`
  - Hypothesis: push the same anchor-relative idea farther out to `h=4` to mimic a more H-JEPA-like substantial future target.
  - Key settings: same as above but `TARGET_FUTURE_STEPS=4`.
  - Artifact: `3,980,950` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=3.528968`, `proxy_val_loss=0.094586`.
  - Decision: discard hard. Far anchor-relative future targets become highly predictable while becoming almost useless for next-byte decoding.

- `pilot_rawctxdelta_ctxlatent_predstack1_predhid1536_dim256_nocontrast_h2_stepwise_5min_g0`
  - Hypothesis: use the `h=2` local future action `z_{t+2}` vs `z_{t+1}` instead of the anchor-relative future state change.
  - Key settings: same as the no-InfoNCE baseline but `TARGET_FUTURE_STEPS=2`, `CONTEXTUAL_DELTA_REFERENCE=stepwise`.
  - Artifact: `3,960,614` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=4.282421`, `proxy_val_loss=0.078831`.
  - Decision: discard hard. Stepwise future deltas are even less aligned with the current-state predictor.

- `pilot_rawctxdelta_ctxlatent_predstack1_predhid1536_dim256_nocontrast_h4_stepwise_5min_g0`
  - Hypothesis: test whether a farther-out local action target works better than the `h=2` stepwise variant.
  - Key settings: same as above but `TARGET_FUTURE_STEPS=4`.
  - Artifact: `3,970,868` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=4.343348`, `proxy_val_loss=0.082769`.
  - Decision: discard hard. Larger stepwise shifts do not help.

- `pilot_rawctxdelta_ctxlatent_predstack1_predhid1536_dim256_mtp2_anchor_nocontrast_rerun_5min_g0`
  - Hypothesis: keep the main `h=1` next-step objective intact and add a single `h=2` anchor-relative auxiliary head, closer to true MTP than replacing the main target.
  - Key settings: matched to the no-InfoNCE baseline but `AUX_TARGET_FUTURE_STEPS=2`, `AUX_TARGET_WEIGHT=0.5`, `CONTEXTUAL_DELTA_REFERENCE=anchor`.
  - Artifact: `4,407,327` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.351834`, `proxy_val_loss=0.364135`.
  - Decision: discard. The auxiliary future anchor head hurts both proxy bpb and artifact size.

- `pilot_rawctxdelta_ctxlatent_predstack1_predhid1536_dim256_mtp2_stepwise_nocontrast_5min_g0`
  - Hypothesis: the auxiliary future head may work better on a local stepwise delta than an anchor-relative one.
  - Key settings: same as the previous auxiliary run but `CONTEXTUAL_DELTA_REFERENCE=stepwise`.
  - Artifact: `4,403,307` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.457626`, `proxy_val_loss=0.298572`.
  - Decision: discard. Stepwise MTP auxiliary supervision is worse again.

## 2026-03-31 Diagonal-Whitened Contextual Delta

- `pilot_rawctxdelta_ctxlatent_predstack1_predhid1536_dim256_nocontrast_h1_anchor_diagwhite_5min_g0`
  - Hypothesis: diagonal-whiten the teacher contextual states before forming the raw orthogonal delta so dominant easy directions are suppressed and the next-byte delta becomes sharper without needing InfoNCE.
  - Key settings: matched to `pilot_rawctxdelta_ctxlatent_predstack1_predhid1536_dim256_nocontrast_h1_anchor_5min_g0` but `CONTEXTUAL_DELTA_WHITENING=diag`, `CONTEXTUAL_DELTA_WHITEN_EPS=1e-4`.
  - Artifact: `3,991,995` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.367952`, `proxy_val_loss=0.315309`.
  - Decision: discard. Diagonal whitening regressed the best no-InfoNCE branch from `2.231334` to `2.367952`.

- `pilot_rawctxdelta_ctxlatent_predstack1_predhid1536_dim256_contrast005_h1_anchor_5min_g0`
  - Hypothesis: establish a matched 5-minute contrastive baseline on the stronger `256d` context-latent predictor branch before testing whitening under the same budget.
  - Key settings: matched to the no-InfoNCE baseline but `CONTRASTIVE_WEIGHT=0.05`, `CONTRASTIVE_MODE=future_state`, `CONTRASTIVE_FUTURE_STEPS=1`.
  - Artifact: `4,048,344` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.419786`, `proxy_val_loss=0.331867`.
  - Decision: discard. At the 5-minute budget on this stronger branch, the contrastive term is already worse than simply training the raw contextual delta with `mse + var/cov`.

- `pilot_rawctxdelta_ctxlatent_predstack1_predhid1536_dim256_contrast005_h1_anchor_diagwhite_5min_g0`
  - Hypothesis: diagonal whitening may interact better with future-state InfoNCE than with the plain no-contrastive objective.
  - Key settings: same as the matched contrastive baseline but `CONTEXTUAL_DELTA_WHITENING=diag`, `CONTEXTUAL_DELTA_WHITEN_EPS=1e-4`.
  - Artifact: `4,043,517` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.438339`, `proxy_val_loss=0.328603`.
  - Decision: discard. Whitening does not rescue the contrastive branch either; it is slightly worse than the already weaker matched contrastive baseline.

## 2026-03-31 Contextual-Delta Sharpening Sweep

- `pilot_rawctxdelta_ctxlatent_predstack1_predhid1536_dim256_span4_nocontrast_5min_g0`
  - Hypothesis: the easy shared context may live in the recent teacher-state subspace, not just along `z_t`, so projecting `z_{t+1}` outside the span of the last `4` teacher states should produce a sharper next-byte delta.
  - Key settings: matched to `pilot_rawctxdelta_ctxlatent_predstack1_predhid1536_dim256_nocontrast_h1_anchor_5min_g0` but `CONTEXTUAL_DELTA_SPAN=4`.
  - Artifact: `4,003,595` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.458743`, `proxy_val_loss=0.277778`.
  - Decision: discard. Removing a recent-state span makes the latent objective easier but hurts next-byte alignment relative to the `2.231334` no-InfoNCE baseline.

- `pilot_rawctxdelta_ctxlatent_predstack1_predhid1536_dim256_topk4_nocontrast_5min_g1`
  - Hypothesis: the raw contextual delta is dominated by a few global teacher PCs, so removing the top `4` principal directions before forming the delta should sharpen the target without needing contrastive pressure.
  - Key settings: matched to `pilot_rawctxdelta_ctxlatent_predstack1_predhid1536_dim256_nocontrast_h1_anchor_5min_g0` but `CONTEXTUAL_DELTA_REMOVE_TOPK=4`.
  - Artifact: `4,023,799` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=3.064657`, `proxy_val_loss=0.194197`.
  - Decision: discard hard. Top-PC removal makes the latent task much easier while destroying token usefulness.

## 2026-04-04 Encoder U-Net Skip Connections

- `smoke_rawctxdelta_ctxlatent_predstack1_predhid1536_dim256_unetskip_10min_g2`
  - Hypothesis: add `train_gpt.py`-style depth-symmetric U-Net skip connections inside the byte-level JEPA encoder so early token-local features flow more directly into deeper contextual states, improving gradient flow and making contextual deltas easier to decode.
  - Key settings: matched to `smoke_rawctxdelta_ctxlatent_predstack1_predhid1536_dim256_10min_g1` but `ENCODER_UNET_SKIP=1`, `ENCODER_UNET_SKIP_INIT=1.0`.
  - Artifact: `4,488,072` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.311050`, `proxy_val_loss=0.251906`.
  - Decision: discard. This regressed sharply versus the current contextual control at `2.117175` while also running slower; the latent objective became easier but the next-byte bridge got worse again.

## 2026-04-04 Scaled 4-Layer U-Net Skip Retry

- `smoke_rawctxdelta_ctxlatent_predstack1_predhid1536_dim256_l4_bs64k_10min_g1`
  - Hypothesis: the shallow `2`-layer encoder only provides one skip pair, so the U-Net wiring may need a deeper encoder to be a fair test; establish the matched `4`-layer contextual-delta control first under the available single-GPU memory budget.
  - Key settings: matched to `smoke_rawctxdelta_ctxlatent_predstack1_predhid1536_dim256_10min_g1` but `NUM_LAYERS=4`, `PREDICTOR_MODEL_LAYERS=4`, `TRAIN_BATCH_BYTES=65536`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  - Artifact: `8,586,199` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.580606`, `proxy_val_loss=0.249431`.
  - Decision: discard as a standalone scaling direction. The deeper model fits and trains cleanly, but the scaled non-skip baseline is much worse than the existing `2`-layer contextual best at `2.117175`.

- `smoke_rawctxdelta_ctxlatent_predstack1_predhid1536_dim256_l4_unetskip05_bs64k_10min_g1`
  - Hypothesis: once the encoder is deep enough to provide multiple skip reinjections, a moderated skip scale should let the U-Net path help the contextual-delta bridge instead of overwhelming it.
  - Key settings: matched to `smoke_rawctxdelta_ctxlatent_predstack1_predhid1536_dim256_l4_bs64k_10min_g1` but `ENCODER_UNET_SKIP=1`, `ENCODER_UNET_SKIP_INIT=0.5`.
  - Artifact: `8,583,304` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.371496`, `proxy_val_loss=0.250811`.
  - Decision: keep as a positive signal only. This is a real improvement over the matched `4`-layer non-skip control (`2.580606 -> 2.371496`) and shows that skips become useful after scaling, but the branch is still not competitive with the older `2`-layer contextual run at `2.117175`.

## 2026-04-04 High-Memory 4-Layer Retry On Empty GPU

- `smoke_rawctxdelta_ctxlatent_predstack1_predhid1536_dim256_l4_bs393216_10min_g0`
  - Hypothesis: the earlier scaled runs were badly underfed by the single-GPU memory cap, so rerunning the `4`-layer contextual-delta branch near the real card limit should give a fairer picture of whether scaling itself helps.
  - Key settings: matched to the `4`-layer control but `TRAIN_BATCH_BYTES=393216` on an otherwise empty GPU, with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  - Artifact: `6,966,364` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.330352`, `proxy_val_loss=0.250255`.
  - Decision: keep as the current best scaled non-skip baseline. This is a large improvement over the constrained `bs64k` scaled control (`2.580606 -> 2.330352`), which confirms the earlier scaled comparison was memory-starved, but it is still behind the older `2`-layer contextual best at `2.117175`.

- `smoke_rawctxdelta_ctxlatent_predstack1_predhid1536_dim256_l4_unetskip05_bs393216_10min_g0`
  - Hypothesis: if scaling was the missing ingredient, the moderated U-Net skip path should benefit once the `4`-layer branch is trained near the real GPU memory limit rather than under the constrained `bs64k` fallback.
  - Key settings: matched to `smoke_rawctxdelta_ctxlatent_predstack1_predhid1536_dim256_l4_bs393216_10min_g0` but `ENCODER_UNET_SKIP=1`, `ENCODER_UNET_SKIP_INIT=0.5`.
  - Artifact: `6,850,815` bytes int8+zlib total.
  - Proxy: `subset_val_bpb=2.385359`, `proxy_val_loss=0.249108`.
  - Decision: discard for now. High-memory scaling helped the whole `4`-layer branch, but the skip model is worse than the matched non-skip control (`2.385359` vs `2.330352`) even though the latent objective is slightly easier again.
