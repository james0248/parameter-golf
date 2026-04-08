# Objective: Pure-JEPA on Byte260

This directory is an isolated non-record Parameter Golf workspace for a byte-level JEPA experiment.

## Hard Rules

- Follow the competition rules in the repo root [AGENTS.md](../../../AGENTS.md).
- Stay in the non-record track unless the user explicitly changes the target.
- Keep the artifact self-contained and under `16_000_000` bytes for any serious candidate run.
- Do not use validation data during training or otherwise leak validation information into the model or artifact.

## Experiment Definition

- The model must be reasonably describable as `pure-JEPA`.
- Byte-level means the experiment uses the `byte260` data/tokenizer family, not SentencePiece or other subword tokenizations.
- If published byte260 shards are unavailable, generate byte260 locally from the published `docs_selected.jsonl` cache rather than switching to subword inputs.
- The training objective must be latent prediction only.
- Do not add next-byte cross-entropy, masked-byte classification, or a hybrid LM auxiliary objective without explicit approval.
- Current special-token policy: control ids (`pad/bos/eos/unk`) may appear in context, but they are excluded from the JEPA target loss and from exact byte-level scoring. The eval distribution is still over the full vocabulary.

## Success Targets

- Minimum target: beat the naive baseline `1.2244 val_bpb`.
- Stronger non-record reference: beat the 4-hour non-record baseline `1.2074 val_bpb`.
- Until the model exposes an exact scoring bridge, results must be labeled as proxy-only.

## Working Boundaries

- Use [train_jepa.py](train_jepa.py) as the stable training skeleton.
- Use [model.py](model.py) for normal experiment iteration.
- Use [SCRATCHPAD.md](SCRATCHPAD.md) as the running decision log.

## Comparison-Only Baseline

- [train_baseline.py](train_baseline.py) and [baseline_gpt.py](baseline_gpt.py) are local comparison tools for measuring a causal LM on `byte260`.
- They do not change the pure-JEPA definition for this workspace and should not be treated as JEPA candidate runs.
