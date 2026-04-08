"""
Byte-level pure-JEPA training scaffold for the Parameter Golf non-record track.

This file intentionally copies the useful generic infrastructure from the root
`train_gpt.py` baseline:
- env-driven hyperparameters
- FineWeb shard loading
- distributed setup
- wallclock-aware training loop
- validation aggregation
- log preservation
- artifact-size accounting with int8 + zlib roundtrip export

Model iteration should normally happen in `model.py`, not in this file.
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

from baseline_gpt import compile_muon_backend
from model import PureByteJEPA, configure_optimizers


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
    val_max_bytes = int(os.environ.get("VAL_MAX_BYTES", 0))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20_000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 0))
    grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", 1))
    train_batch_bytes = int(os.environ.get("TRAIN_BATCH_BYTES", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    enable_torch_compile = bool(int(os.environ.get("ENABLE_TORCH_COMPILE", "0")))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 260))
    byte_offset = int(os.environ.get("BYTE_OFFSET", 4))
    byte_count = int(os.environ.get("BYTE_COUNT", 256))
    allow_nonstandard_byte_data = bool(int(os.environ.get("ALLOW_NONSTANDARD_BYTE_DATA", "0")))

    model_dim = int(os.environ.get("MODEL_DIM", 256))
    num_layers = int(os.environ.get("NUM_LAYERS", 4))
    num_heads = int(os.environ.get("NUM_HEADS", 4))
    mlp_hidden_dim = int(os.environ.get("MLP_HIDDEN_DIM", 1024))
    predictor_hidden_dim = int(os.environ.get("PREDICTOR_HIDDEN_DIM", 512))
    predictor_num_layers = int(os.environ.get("PREDICTOR_NUM_LAYERS", 0))
    predictor_num_heads = int(os.environ.get("PREDICTOR_NUM_HEADS", os.environ.get("NUM_HEADS", "4")))
    predictor_input = os.environ.get("PREDICTOR_INPUT", "context_latent")
    predictor_model_layers = int(os.environ.get("PREDICTOR_MODEL_LAYERS", os.environ.get("NUM_LAYERS", "4")))
    predictor_model_heads = int(os.environ.get("PREDICTOR_MODEL_HEADS", os.environ.get("NUM_HEADS", "4")))
    predictor_model_mlp_hidden_dim = int(
        os.environ.get("PREDICTOR_MODEL_MLP_HIDDEN_DIM", os.environ.get("MLP_HIDDEN_DIM", "1024"))
    )
    dropout = float(os.environ.get("DROPOUT", 0.0))
    target_ema_decay = float(os.environ.get("TARGET_EMA_DECAY", 0.99))
    eval_logit_scale = float(os.environ.get("EVAL_LOGIT_SCALE", 1.0))
    jepa_loss = os.environ.get("JEPA_LOSS", "cosine")
    eval_bridge = os.environ.get("EVAL_BRIDGE", "cosine")
    target_codebook = os.environ.get("TARGET_CODEBOOK", "ema")
    num_target_codebooks = int(os.environ.get("NUM_TARGET_CODEBOOKS", 1))
    contextual_target = bool(int(os.environ.get("CONTEXTUAL_TARGET", "0")))
    target_source = os.environ.get("TARGET_SOURCE", "codebook")
    target_encoder_causal = bool(int(os.environ.get("TARGET_ENCODER_CAUSAL", "0")))
    target_anchor_weight = float(os.environ.get("TARGET_ANCHOR_WEIGHT", 0.5))
    prototype_ema_decay = float(os.environ.get("PROTOTYPE_EMA_DECAY", 0.99))
    contextual_residual_scale = float(os.environ.get("CONTEXTUAL_RESIDUAL_SCALE", 0.0))
    contextual_aux_weight = float(os.environ.get("CONTEXTUAL_AUX_WEIGHT", 0.0))
    lm_probe_weight = float(os.environ.get("LM_PROBE_WEIGHT", 0.0))
    lm_probe_input = os.environ.get("LM_PROBE_INPUT", "pred_target")
    split_lm_probe_optimizer = bool(int(os.environ.get("SPLIT_LM_PROBE_OPTIMIZER", "0")))
    lm_probe_lr = float(os.environ.get("LM_PROBE_LR", os.environ.get("LR", "3e-4")))
    optimizer = os.environ.get("OPTIMIZER", "adamw")
    embed_lr = float(os.environ.get("EMBED_LR", os.environ.get("LR", "3e-4")))
    matrix_lr = float(os.environ.get("MATRIX_LR", os.environ.get("LR", "3e-4")))
    scalar_lr = float(os.environ.get("SCALAR_LR", os.environ.get("LR", "3e-4")))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    var_reg_weight = float(os.environ.get("VAR_REG_WEIGHT", 0.0))
    cov_reg_weight = float(os.environ.get("COV_REG_WEIGHT", 0.0))

    lr = float(os.environ.get("LR", 3e-4))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.01))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))


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


def hyperparameters_dict(args: Hyperparameters) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in vars(args.__class__).items():
        if name.startswith("_") or callable(value):
            continue
        out[name] = getattr(args, name)
    return out


def metric_tensor(value: Tensor | float | int, device: torch.device, dtype: torch.dtype = torch.float64) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(float(value), device=device, dtype=dtype)


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


def validate_byte_tensor(values: Tensor, *, args: Hyperparameters, source: str) -> None:
    if values.numel() == 0:
        raise ValueError(f"{source} is empty")
    check_values = values if values.dtype != torch.uint16 else values.to(dtype=torch.int32)
    min_id = int(check_values.min().item())
    max_id = int(check_values.max().item())
    if min_id < 0 or max_id >= args.vocab_size:
        raise ValueError(
            f"{source} contains ids outside the byte260 tokenizer range "
            f"[0, {args.vocab_size - 1}]: min={min_id} max={max_id}"
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
    values_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if values_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    values = torch.from_numpy(values_np.astype(np.uint16, copy=False))
    validate_byte_tensor(values, args=args, source=str(file))
    return values


class ByteStream:
    def __init__(self, pattern: str, *, args: Hyperparameters):
        self.args = args
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.values = load_data_shard(self.files[0], args=args)
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.values = load_data_shard(self.files[self.file_idx], args=self.args)
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            available = self.values.numel() - self.pos
            if available <= 0:
                self._advance_file()
                continue
            k = min(remaining, available)
            chunks.append(self.values[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedByteLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device, *, args: Hyperparameters):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.args = args
        self.stream = ByteStream(pattern, args=args)

    def next_batch(self, global_bytes: int, seq_len: int, grad_accum_steps: int) -> Tensor:
        denom = self.world_size * grad_accum_steps
        if global_bytes % denom != 0:
            raise ValueError(f"TRAIN_BATCH_BYTES={global_bytes} must be divisible by WORLD_SIZE*GRAD_ACCUM_STEPS={denom}")
        local_bytes = global_bytes // denom
        if local_bytes % seq_len != 0:
            raise ValueError(f"Per-rank microbatch bytes={local_bytes} must be divisible by TRAIN_SEQ_LEN={seq_len}")
        chunk = self.stream.take(local_bytes * self.world_size)
        start = self.rank * local_bytes
        local = chunk[start : start + local_bytes].to(dtype=torch.int64)
        return local.reshape(-1, seq_len).to(self.device, non_blocking=True)


def load_validation_bytes(pattern: str, seq_len: int, *, args: Hyperparameters) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    values = torch.cat([load_data_shard(file, args=args) for file in files]).contiguous()
    usable = (values.numel() // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return values[:usable]


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern for pattern in os.environ.get("CONTROL_TENSOR_NAME_PATTERNS", "").split(",") if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get("INT8_KEEP_FLOAT_FP32_NAME_PATTERNS", ",".join(CONTROL_TENSOR_NAME_PATTERNS)).split(",")
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


def eval_validation(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_bytes: Tensor,
    *,
    full_validation: bool,
) -> dict[str, float | None]:
    if args.val_batch_size % world_size != 0:
        raise ValueError(f"VAL_BATCH_SIZE={args.val_batch_size} must be divisible by WORLD_SIZE={world_size}")
    local_batch_bytes = args.val_batch_size // world_size
    if local_batch_bytes < args.train_seq_len:
        raise ValueError(
            f"VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    if local_batch_bytes % args.train_seq_len != 0:
        raise ValueError(f"Per-rank validation bytes={local_batch_bytes} must be divisible by TRAIN_SEQ_LEN={args.train_seq_len}")

    local_batch_seqs = local_batch_bytes // args.train_seq_len
    total_seqs = val_bytes.numel() // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    nll_sum = torch.zeros((), device=device, dtype=torch.float64)
    batch_count = torch.zeros((), device=device, dtype=torch.int64)
    nll_batch_count = torch.zeros((), device=device, dtype=torch.int64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len
            batch = val_bytes[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True).reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_metrics = model(batch, return_validation=True)
            loss_sum += metric_tensor(batch_metrics["loss_sum"], device)
            byte_count += metric_tensor(batch_metrics["num_bytes"], device)
            batch_count += 1
            if "nll_sum_nat" in batch_metrics:
                nll_sum += metric_tensor(batch_metrics["nll_sum_nat"], device)
                nll_batch_count += 1

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(nll_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(batch_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(nll_batch_count, op=dist.ReduceOp.SUM)

    if byte_count.item() <= 0:
        raise ValueError("Validation byte count must be positive")

    proxy_loss = float((loss_sum / byte_count).item())
    has_exact_nll = int(batch_count.item()) > 0 and int(nll_batch_count.item()) == int(batch_count.item())
    subset_val_bpb = float((nll_sum / (math.log(2.0) * byte_count)).item()) if has_exact_nll else None
    val_bpb = subset_val_bpb if full_validation else None
    model.train()
    return {
        "proxy_loss": proxy_loss,
        "subset_val_bpb": subset_val_bpb,
        "val_bpb": val_bpb,
    }


def main() -> None:
    code = Path(__file__).read_text(encoding="utf-8")
    model_code = (SCRIPT_DIR / "model.py").read_text(encoding="utf-8")
    args = Hyperparameters()
    compile_muon_backend(enabled=args.enable_torch_compile and args.optimizer.lower() == "muon")

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if args.grad_accum_steps <= 0:
        raise ValueError(f"GRAD_ACCUM_STEPS must be positive, got {args.grad_accum_steps}")
    grad_scale = 1.0 / args.grad_accum_steps

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
        (run_dir / "train_jepa_snapshot.py").write_text(code, encoding="utf-8")
        (run_dir / "model_snapshot.py").write_text(model_code, encoding="utf-8")

    log0(code, console=False)
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
    val_bytes = load_validation_bytes(args.val_files, args.train_seq_len, args=args)
    full_val_bytes = int(val_bytes.numel())
    val_is_capped = False
    if args.val_max_bytes > 0 and full_val_bytes > args.val_max_bytes:
        capped_bytes = (args.val_max_bytes // args.train_seq_len) * args.train_seq_len
        if capped_bytes <= 0:
            raise ValueError(
                f"VAL_MAX_BYTES={args.val_max_bytes} is too small for TRAIN_SEQ_LEN={args.train_seq_len}"
            )
        val_bytes = val_bytes[:capped_bytes].contiguous()
        val_is_capped = True
    log0(f"dataset_kind:pure_byte tokenizer_path:{args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(
        f"val_loader:shards pattern={args.val_files} bytes:{val_bytes.numel()} "
        f"full_bytes:{full_val_bytes} subset_only:{int(val_is_capped)}"
    )

    base_model = PureByteJEPA(
        vocab_size=args.vocab_size,
        byte_offset=args.byte_offset,
        byte_count=args.byte_count,
        seq_len=args.train_seq_len,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_hidden_dim=args.mlp_hidden_dim,
        predictor_hidden_dim=args.predictor_hidden_dim,
        predictor_num_layers=args.predictor_num_layers,
        predictor_num_heads=args.predictor_num_heads,
        predictor_input=args.predictor_input,
        predictor_model_layers=args.predictor_model_layers,
        predictor_model_heads=args.predictor_model_heads,
        predictor_model_mlp_hidden_dim=args.predictor_model_mlp_hidden_dim,
        dropout=args.dropout,
        target_ema_decay=args.target_ema_decay,
        eval_logit_scale=args.eval_logit_scale,
        jepa_loss=args.jepa_loss,
        eval_bridge=args.eval_bridge,
        target_codebook=args.target_codebook,
        num_target_codebooks=args.num_target_codebooks,
        contextual_target=args.contextual_target,
        target_source=args.target_source,
        target_encoder_causal=args.target_encoder_causal,
        target_anchor_weight=args.target_anchor_weight,
        prototype_ema_decay=args.prototype_ema_decay,
        contextual_residual_scale=args.contextual_residual_scale,
        contextual_aux_weight=args.contextual_aux_weight,
        lm_probe_weight=args.lm_probe_weight,
        lm_probe_input=args.lm_probe_input,
        var_reg_weight=args.var_reg_weight,
        cov_reg_weight=args.cov_reg_weight,
    ).to(device)
    compiled_or_base: nn.Module = torch.compile(base_model, dynamic=False) if args.enable_torch_compile else base_model
    model: nn.Module = DDP(compiled_or_base, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_or_base

    optimizers = configure_optimizers(base_model, args)
    if not isinstance(optimizers, list):
        optimizers = [optimizers]
    for opt in optimizers:
        for group in opt.param_groups:
            group["base_lr"] = group.get("lr", args.lr)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{args.grad_accum_steps}")
    log0(
        f"train_batch_bytes:{args.train_batch_bytes} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(
        f"optimizer:{args.optimizer} embed_lr:{args.embed_lr:.6g} matrix_lr:{args.matrix_lr:.6g} "
        f"scalar_lr:{args.scalar_lr:.6g} lm_probe_lr:{args.lm_probe_lr:.6g}"
    )
    log0(f"seed:{args.seed}")

    train_loader = DistributedByteLoader(args.train_files, rank, world_size, device, args=args)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    def optimizer_parameters(opt: torch.optim.Optimizer) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        seen: set[int] = set()
        for group in opt.param_groups:
            for param in group["params"]:
                if not isinstance(param, nn.Parameter) or not param.requires_grad or id(param) in seen:
                    continue
                seen.add(id(param))
                params.append(param)
        return params

    optimizer_param_sets = [optimizer_parameters(opt) for opt in optimizers]

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
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
            for micro_step in range(args.grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == args.grad_accum_steps - 1
                batch = train_loader.next_batch(args.train_batch_bytes, args.train_seq_len, args.grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_metrics = model(batch)
                (warmup_metrics["loss"] * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            base_model.update_target_encoder()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        train_loader = DistributedByteLoader(args.train_files, rank, world_size, device, args=args)

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
            val_metrics = eval_validation(
                args,
                model,
                rank,
                world_size,
                device,
                val_bytes,
                full_validation=not val_is_capped,
            )
            if val_metrics["val_bpb"] is None:
                subset_bpb = val_metrics["subset_val_bpb"]
                subset_msg = f"val_bpb_subset:{subset_bpb:.6f} " if subset_bpb is not None else "val_bpb_subset:NA "
                log0(
                    f"step:{step}/{args.iterations} val_proxy_loss:{val_metrics['proxy_loss']:.6f} "
                    f"{subset_msg}val_bpb:NA proxy_only:1 train_time:{training_time_ms:.0f}ms "
                    f"step_avg:{training_time_ms / max(step, 1):.2f}ms"
                )
            else:
                log0(
                    f"step:{step}/{args.iterations} val_proxy_loss:{val_metrics['proxy_loss']:.6f} "
                    f"val_bpb:{val_metrics['val_bpb']:.6f} train_time:{training_time_ms:.0f}ms "
                    f"step_avg:{training_time_ms / max(step, 1):.2f}ms"
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
        train_loss = torch.zeros((), device=device, dtype=torch.float64)
        for micro_step in range(args.grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == args.grad_accum_steps - 1
            batch = train_loader.next_batch(args.train_batch_bytes, args.train_seq_len, args.grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                metrics = model(batch)
            loss = metric_tensor(metrics["loss"], device, dtype=torch.float32)
            train_loss += loss.detach().to(dtype=torch.float64)
            (loss * grad_scale).backward()
        train_loss /= args.grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
                if "momentum" in group:
                    group["momentum"] = muon_momentum

        if args.grad_clip_norm > 0:
            for params in optimizer_param_sets:
                if params:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        base_model.update_target_encoder()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.6f} "
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

    export_state = base_model.export_state_dict() if hasattr(base_model, "export_state_dict") else base_model.state_dict()
    raw_model_path = run_dir / "final_model.pt"
    quant_model_path = run_dir / "final_model.int8.ptz"
    summary_path = run_dir / "summary.json"

    if master_process:
        torch.save(export_state, raw_model_path)
        model_bytes = raw_model_path.stat().st_size
        code_bytes = len(code.encode("utf-8")) + len(model_code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(export_state)
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with quant_model_path.open("wb") as f:
            f.write(quant_blob)
        quant_file_bytes = quant_model_path.stat().st_size
        code_bytes = len(code.encode("utf-8")) + len(model_code.encode("utf-8"))
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
    load_result = base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=False)
    if master_process and (load_result.missing_keys or load_result.unexpected_keys):
        log0(
            f"roundtrip_state_warnings missing_keys:{list(load_result.missing_keys)} "
            f"unexpected_keys:{list(load_result.unexpected_keys)}"
        )

    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_metrics = eval_validation(
        args,
        model,
        rank,
        world_size,
        device,
        val_bytes,
        full_validation=not val_is_capped,
    )
    torch.cuda.synchronize()
    eval_ms = 1000.0 * (time.perf_counter() - t_qeval)
    if q_val_metrics["val_bpb"] is None:
        subset_bpb = q_val_metrics["subset_val_bpb"]
        subset_msg = f"val_bpb_subset:{subset_bpb:.6f} " if subset_bpb is not None else "val_bpb_subset:NA "
        log0(
            f"final_int8_zlib_roundtrip val_proxy_loss:{q_val_metrics['proxy_loss']:.6f} "
            f"{subset_msg}val_bpb:NA proxy_only:1 eval_time:{eval_ms:.0f}ms"
        )
    else:
        log0(
            f"final_int8_zlib_roundtrip val_proxy_loss:{q_val_metrics['proxy_loss']:.6f} "
            f"val_bpb:{q_val_metrics['val_bpb']:.8f} eval_time:{eval_ms:.0f}ms"
        )

    if master_process:
        summary = {
            "run_id": args.run_id,
            "proxy_val_loss": q_val_metrics["proxy_loss"],
            "subset_val_bpb": q_val_metrics["subset_val_bpb"],
            "val_bpb": q_val_metrics["val_bpb"],
            "proxy_only": q_val_metrics["val_bpb"] is None,
            "validation_bytes": int(val_bytes.numel()),
            "validation_full_bytes": full_val_bytes,
            "artifact_model_bytes_int8_zlib": quant_model_path.stat().st_size,
            "artifact_code_bytes": len(code.encode("utf-8")) + len(model_code.encode("utf-8")),
            "artifact_total_bytes_int8_zlib": quant_model_path.stat().st_size + len(code.encode("utf-8")) + len(model_code.encode("utf-8")),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
