"""
Byte-level GPT baseline trainer for the Pure-JEPA workspace.

This is a comparison-only baseline on `byte260`. It keeps the root baseline
architecture and optimizer split, but evaluates exact byte-level `val_bpb`
by masking out control-token targets such as BOS.
"""

from __future__ import annotations

import copy
import glob
import io
import json
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from baseline_gpt import (
    CONTROL_TENSOR_NAME_PATTERNS,
    CastedLinear,
    GPT,
    compile_muon_backend,
    configure_optimizers,
    restore_low_dim_params_to_fp32,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
REPO_DATASET_PATH = REPO_ROOT / "data" / "datasets" / "fineweb10B_byte260"
REPO_TOKENIZER_PATH = REPO_ROOT / "data" / "tokenizers" / "fineweb_pure_byte_260.json"


class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", str(REPO_DATASET_PATH))
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", str(REPO_TOKENIZER_PATH))
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20_000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    enable_torch_compile = bool(int(os.environ.get("ENABLE_TORCH_COMPILE", "1")))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 260))
    byte_offset = int(os.environ.get("BYTE_OFFSET", 4))
    byte_count = int(os.environ.get("BYTE_COUNT", 256))
    allow_nonstandard_byte_data = bool(int(os.environ.get("ALLOW_NONSTANDARD_BYTE_DATA", "0")))

    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))


@dataclass(frozen=True)
class PureByteTokenizerConfig:
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3
    byte_offset: int = 4
    byte_count: int = 256

    @property
    def vocab_size(self) -> int:
        return self.byte_offset + self.byte_count

    @property
    def byte_token_max(self) -> int:
        return self.byte_offset + self.byte_count


def hyperparameters_dict(args: Hyperparameters) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in vars(args.__class__).items():
        if name.startswith("_") or callable(value):
            continue
        out[name] = getattr(args, name)
    return out


def load_byte_tokenizer_config(path: str) -> PureByteTokenizerConfig:
    tokenizer_path = Path(path)
    if not tokenizer_path.is_file():
        return PureByteTokenizerConfig()
    payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    if payload.get("tokenizer_type") != "pure_byte":
        raise ValueError(f"Expected a pure_byte tokenizer JSON, got {payload.get('tokenizer_type')!r}")
    config = payload.get("config", {})
    tokenizer = PureByteTokenizerConfig(
        pad_id=int(config.get("pad_id", 0)),
        bos_id=int(config.get("bos_id", 1)),
        eos_id=int(config.get("eos_id", 2)),
        unk_id=int(config.get("unk_id", 3)),
        byte_offset=int(config.get("byte_offset", 4)),
        byte_count=int(config.get("byte_count", 256)),
    )
    vocab_size = int(payload.get("vocab_size", tokenizer.vocab_size))
    if vocab_size != tokenizer.vocab_size:
        raise ValueError(f"Tokenizer vocab_size mismatch: json={vocab_size} config={tokenizer.vocab_size}")
    return tokenizer


def validate_token_tensor(values: Tensor, *, args: Hyperparameters, source: str) -> None:
    if values.numel() == 0:
        raise ValueError(f"{source} is empty")
    check_values = values if values.dtype != torch.uint16 else values.to(dtype=torch.int32)
    min_id = int(check_values.min().item())
    max_id = int(check_values.max().item())
    if min_id < 0 or max_id >= args.vocab_size:
        raise ValueError(
            f"{source} contains ids outside the tokenizer range [0, {args.vocab_size - 1}]: "
            f"min={min_id} max={max_id}"
        )


def load_data_shard(file: Path, *, args: Hyperparameters) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    values = torch.from_numpy(tokens_np.astype(np.uint16, copy=False))
    validate_token_tensor(values, args=args, source=str(file))
    return values


class TokenStream:
    def __init__(self, pattern: str, *, args: Hyperparameters):
        self.args = args
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0], args=args)
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx], args=self.args)
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            available = self.tokens.numel() - self.pos
            if available <= 0:
                self._advance_file()
                continue
            k = min(remaining, available)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device, *, args: Hyperparameters):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.args = args
        self.stream = TokenStream(pattern, args=args)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        denom = self.world_size * grad_accum_steps
        if global_tokens % denom != 0:
            raise ValueError(f"TRAIN_BATCH_TOKENS={global_tokens} must be divisible by WORLD_SIZE*GRAD_ACCUM_STEPS={denom}")
        local_tokens = global_tokens // denom
        if local_tokens % seq_len != 0:
            raise ValueError(f"Per-rank microbatch tokens={local_tokens} must be divisible by TRAIN_SEQ_LEN={seq_len}")
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


def load_validation_tokens(pattern: str, seq_len: int, *, args: Hyperparameters) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file, args=args) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict_int8(state_dict: dict[str, Tensor]) -> tuple[dict[str, object], dict[str, int]]:
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    base_model: GPT,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    tokenizer: PureByteTokenizerConfig,
) -> dict[str, float]:
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    if local_batch_tokens % args.train_seq_len != 0:
        raise ValueError(
            f"Per-rank validation tokens={local_batch_tokens} must be divisible by TRAIN_SEQ_LEN={args.train_seq_len}"
        )

    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    payload_nll_sum = torch.zeros((), device=device, dtype=torch.float64)
    payload_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    control_target_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    base_model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_nll = base_model(x, y, reduction="none")
            flat_nll = batch_nll.reshape(-1).to(dtype=torch.float64)
            flat_targets = y.reshape(-1)
            payload_mask = (flat_targets >= tokenizer.byte_offset) & (flat_targets < tokenizer.byte_token_max)

            val_loss_sum += flat_nll.sum()
            val_token_count += float(flat_targets.numel())
            payload_nll_sum += flat_nll[payload_mask].sum()
            payload_byte_count += payload_mask.to(dtype=torch.float64).sum()
            control_target_count += (~payload_mask).to(dtype=torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(payload_nll_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(payload_byte_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(control_target_count, op=dist.ReduceOp.SUM)

    if payload_byte_count.item() <= 0:
        raise ValueError("Validation payload byte count must be positive")

    val_loss_all = float((val_loss_sum / val_token_count).item())
    val_loss_payload = float((payload_nll_sum / payload_byte_count).item())
    val_bpb = float((payload_nll_sum / (math.log(2.0) * payload_byte_count)).item())
    control_target_frac = float((control_target_count / val_token_count).item())
    model.train()
    base_model.train()
    return {
        "val_loss_all": val_loss_all,
        "val_loss_payload": val_loss_payload,
        "val_bpb": val_bpb,
        "control_target_frac": control_target_frac,
    }


def main() -> None:
    train_code = Path(__file__).read_text(encoding="utf-8")
    model_code = (SCRIPT_DIR / "baseline_gpt.py").read_text(encoding="utf-8")
    args = Hyperparameters()
    compile_muon_backend(enabled=args.enable_torch_compile)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training scaffold")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    run_dir = SCRIPT_DIR / "runs" / args.run_id
    logfile = run_dir / "train.log"
    if master_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    if master_process:
        (run_dir / "config.json").write_text(json.dumps(hyperparameters_dict(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (run_dir / "train_baseline_snapshot.py").write_text(train_code, encoding="utf-8")
        (run_dir / "baseline_gpt_snapshot.py").write_text(model_code, encoding="utf-8")

    log0(train_code, console=False)
    log0("=" * 100, console=False)
    log0(model_code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = load_byte_tokenizer_config(args.tokenizer_path)
    dataset_dir = Path(args.data_path).resolve()
    if not args.allow_nonstandard_byte_data and "byte260" not in dataset_dir.name:
        raise ValueError(
            f"Expected a byte260 dataset directory by default, got {dataset_dir.name!r}. "
            "Set ALLOW_NONSTANDARD_BYTE_DATA=1 only if you are intentionally using a compatible custom export."
        )
    if tokenizer.vocab_size != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={tokenizer.vocab_size}")
    if tokenizer.byte_offset != args.byte_offset or tokenizer.byte_count != args.byte_count:
        raise ValueError(
            f"Byte tokenizer config mismatch: offset/count={tokenizer.byte_offset}/{tokenizer.byte_count}, "
            f"expected {args.byte_offset}/{args.byte_count}"
        )

    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len, args=args)
    log0(f"val_bpb:enabled tokenizer_kind=pure_byte payload_only=1 tokenizer_path:{args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model: nn.Module = (
        torch.compile(base_model, dynamic=False, fullgraph=True) if args.enable_torch_compile else base_model
    )
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    optimizers = configure_optimizers(base_model, args)
    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{args.tied_embed_lr if args.tie_embeddings else args.embed_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device, args=args)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            if warmdown_start <= step < args.iterations:
                return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device, args=args)

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_metrics = eval_val(args, model, base_model, rank, world_size, device, grad_accum_steps, val_tokens, tokenizer)
            log0(
                f"step:{step}/{args.iterations} val_loss_all:{val_metrics['val_loss_all']:.4f} "
                f"val_loss_payload:{val_metrics['val_loss_payload']:.4f} val_bpb:{val_metrics['val_bpb']:.4f} "
                f"control_target_frac:{val_metrics['control_target_frac']:.6f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
                if "momentum" in group:
                    group["momentum"] = muon_momentum

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    raw_model_path = run_dir / "final_model.pt"
    quant_model_path = run_dir / "final_model.int8.ptz"
    summary_path = run_dir / "summary.json"

    if master_process:
        torch.save(base_model.state_dict(), raw_model_path)
        model_bytes = raw_model_path.stat().st_size
        code_bytes = len(train_code.encode("utf-8")) + len(model_code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with quant_model_path.open("wb") as f:
            f.write(quant_blob)
        quant_file_bytes = quant_model_path.stat().st_size
        code_bytes = len(train_code.encode("utf-8")) + len(model_code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()

    with quant_model_path.open("rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)

    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_metrics = eval_val(args, model, base_model, rank, world_size, device, grad_accum_steps, val_tokens, tokenizer)
    torch.cuda.synchronize()
    eval_ms = 1000.0 * (time.perf_counter() - t_qeval)
    log0(
        f"final_int8_zlib_roundtrip val_loss_all:{q_val_metrics['val_loss_all']:.4f} "
        f"val_loss_payload:{q_val_metrics['val_loss_payload']:.4f} val_bpb:{q_val_metrics['val_bpb']:.4f} "
        f"control_target_frac:{q_val_metrics['control_target_frac']:.6f} eval_time:{eval_ms:.0f}ms"
    )
    log0(
        f"final_int8_zlib_roundtrip_exact val_loss_all:{q_val_metrics['val_loss_all']:.8f} "
        f"val_loss_payload:{q_val_metrics['val_loss_payload']:.8f} val_bpb:{q_val_metrics['val_bpb']:.8f}"
    )

    if master_process:
        summary = {
            "run_id": args.run_id,
            "val_loss_all_targets": q_val_metrics["val_loss_all"],
            "val_loss_payload_only": q_val_metrics["val_loss_payload"],
            "val_bpb": q_val_metrics["val_bpb"],
            "control_target_frac": q_val_metrics["control_target_frac"],
            "artifact_model_bytes_int8_zlib": quant_model_path.stat().st_size,
            "artifact_code_bytes": len(train_code.encode("utf-8")) + len(model_code.encode("utf-8")),
            "artifact_total_bytes_int8_zlib": quant_model_path.stat().st_size + len(train_code.encode("utf-8")) + len(model_code.encode("utf-8")),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
