# Final Pure-Byte Commands

Run from repo root: `/home/clp/hyeonseok/parameter-golf`.

This is the single final branch: pure byte260, no contrastive loss, no SmearGate, no added token vocabulary, lag mixer enabled with BOS masking, and the target-pruned mixed int8 export.

Fresh 8xH100 machine setup, data download, 10-minute-capped training, and submission bundle:

```bash
SUBMISSION_AUTHOR="Your Name" SUBMISSION_GITHUB_ID="your-github" \
records/track_non_record_16mb/2026-05-01_PureByteJEPA_FinalWrap/run_fresh_h100.sh
```

Direct trainer command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
RUN_ID=finalwrap_lagmixer2_pruned_mixed_10min_gpu0 \
MAX_WALLCLOCK_SECONDS=600 ITERATIONS=20000 WARMDOWN_ITERS=1200 VAL_MAX_BYTES=0 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=50 \
/opt/miniconda3/envs/base5090/bin/python records/track_non_record_16mb/2026-05-01_PureByteJEPA_FinalWrap/train_jepa.py
```

Sanity-check by adding `DRY_RUN=1`; the script validates the final-branch constraints without starting CUDA training.
