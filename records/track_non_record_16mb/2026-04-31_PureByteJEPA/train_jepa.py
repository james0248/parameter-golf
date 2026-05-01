#!/usr/bin/env python3
"""Final-only pure-byte JEPA trainer.

Kept branch: byte260, contextual-delta JEPA, LagMixer, auxiliary LM head,
8xH100 DDP Muon training, train_gpt-style contiguous validation, mixed int8 export.
"""

from __future__ import annotations

import glob
import io
import json
import math
import os
import random
import time
import uuid
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]


@dataclass
class Cfg:
    data_path: str = os.environ.get("DATA_PATH", str(REPO_ROOT / "data/datasets/fineweb10B_byte260"))
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", str(REPO_ROOT / "data/tokenizers/fineweb_pure_byte_260.json"))
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed: int = int(os.environ.get("SEED", "1337"))
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", "524288"))
    val_max_bytes: int = int(os.environ.get("VAL_MAX_BYTES", "0"))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", "0"))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", "500"))
    iterations: int = int(os.environ.get("ITERATIONS", "20000"))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", "1200"))
    warmdown_frac: float = float(os.environ.get("WARMDOWN_FRAC", "0.85"))
    train_batch_bytes: int = int(os.environ.get("TRAIN_BATCH_BYTES", "786432"))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", "256"))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", "0"))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", "600"))
    vocab_size: int = 260
    bos_token_id: int = 1
    byte_offset: int = 4
    byte_count: int = 256
    model_dim: int = 384
    num_layers: int = 2
    num_heads: int = 8
    mlp_hidden_dim: int = 1536
    predictor_hidden_dim: int = 2048
    predictor_num_layers: int = 1
    lm_probe_hidden_dim: int = 1024
    target_ema_decay: float = float(os.environ.get("TARGET_EMA_DECAY", "0.9965"))
    lm_probe_weight: float = 0.5
    var_reg_weight: float = 1.0
    cov_reg_weight: float = 0.0225
    min_lr_scale: float = float(os.environ.get("MIN_LR_SCALE", os.environ.get("MIN_LR", "0.10")))
    embed_lr: float = float(os.environ.get("EMBED_LR", "0.03"))
    matrix_lr: float = float(os.environ.get("MATRIX_LR", "0.026"))
    scalar_lr: float = float(os.environ.get("SCALAR_LR", "0.02"))
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = float(os.environ.get("BETA2", "0.99"))
    adam_eps: float = 1e-8
    grad_clip_norm: float = 0.3
    muon_momentum: float = float(os.environ.get("MUON_MOMENTUM", "0.97"))
    muon_backend_steps: int = 5
    muon_momentum_warmup_start: float = 0.92
    muon_momentum_warmup_steps: int = 1500
    lag_mixer_lags: int = 2
    lag_mixer_init: float = 0.1
    bos_attention_mask: bool = False

    @property
    def train_files(self) -> str:
        return str(Path(self.data_path) / "fineweb_train_*.bin")

    @property
    def val_files(self) -> str:
        return str(Path(self.data_path) / "fineweb_val_*.bin")


def zeropower_via_newtonschulz5(g: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.bfloat16()
    x /= x.norm() + eps
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        aa = x @ x.T
        x = a * x + (b * aa + c * aa @ aa) @ x
    return x.T if transposed else x


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            total = sum(p.numel() for p in params)
            flat = torch.zeros(total, device=params[0].device, dtype=torch.bfloat16)
            pos = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    state = self.state[p]
                    buf = state.setdefault("momentum_buffer", torch.zeros_like(p.grad))
                    buf.mul_(group["momentum"]).add_(p.grad)
                    g = p.grad.add(buf, alpha=group["momentum"])
                    g = zeropower_via_newtonschulz5(g, group["backend_steps"])
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    flat[pos : pos + p.numel()] = g.reshape(-1)
                pos += p.numel()
            if distributed:
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            pos = 0
            for p in params:
                p.add_(flat[pos : pos + p.numel()].view_as(p).to(p.dtype), alpha=-group["lr"])
                pos += p.numel()
        return loss


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, hidden: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class LagMixer(nn.Module):
    def __init__(self, dim: int, lags: int, init: float, bos_token_id: int):
        super().__init__()
        self.lags = lags
        self.bos_token_id = bos_token_id
        self.lag_weights = nn.Parameter(torch.full((lags, dim), init / lags, dtype=torch.float32))

    def _mask(self, tokens: Tensor, lag: int) -> Tensor:
        seq = tokens.size(1)
        pos = torch.arange(seq, device=tokens.device).view(1, seq)
        last_bos = torch.cummax(torch.where(tokens == self.bos_token_id, pos, torch.zeros_like(pos)), dim=1).values
        return (pos - lag >= last_bos) & (tokens != self.bos_token_id)

    def forward(self, x: Tensor, tokens: Tensor) -> Tensor:
        mixed = torch.zeros_like(x)
        for lag in range(1, min(self.lags, x.size(1) - 1) + 1):
            shifted = torch.zeros_like(x)
            shifted[:, lag:] = x[:, :-lag]
            shifted = shifted * self._mask(tokens, lag).unsqueeze(-1).to(shifted.dtype)
            mixed = mixed + self.lag_weights[lag - 1].to(x.dtype)[None, None, :] * shifted
        return x + mixed


class SequenceEncoder(nn.Module):
    def __init__(self, cfg: Cfg, *, causal: bool):
        super().__init__()
        self.causal = causal
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.train_seq_len + 1, cfg.model_dim))
        self.lag_mixer = LagMixer(cfg.model_dim, cfg.lag_mixer_lags, cfg.lag_mixer_init, cfg.bos_token_id)
        self.blocks = nn.ModuleList(
            [EncoderBlock(cfg.model_dim, cfg.num_heads, cfg.mlp_hidden_dim) for _ in range(cfg.num_layers)]
        )
        self.norm = nn.LayerNorm(cfg.model_dim)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.01)

    def _attention_mask(self, tokens: Tensor) -> Tensor | None:
        if not self.causal:
            return None
        seq = tokens.size(1)
        pos = torch.arange(seq, device=tokens.device)
        causal = pos.view(1, seq) > pos.view(seq, 1)
        if not self.cfg.bos_attention_mask:
            return causal
        batch_pos = pos.view(1, seq)
        last_bos = torch.cummax(torch.where(tokens == self.cfg.bos_token_id, batch_pos, torch.zeros_like(batch_pos)), dim=1).values
        doc = causal.unsqueeze(0) | (pos.view(1, 1, seq) < last_bos.unsqueeze(-1))
        return doc.repeat_interleave(self.cfg.num_heads, dim=0)

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.tok_emb(tokens) + self.pos_emb[:, : tokens.size(1)]
        x = self.lag_mixer(x, tokens)
        mask = self._attention_mask(tokens)
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)


class Predictor(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.blocks = nn.ModuleList(
            [EncoderBlock(cfg.model_dim, cfg.num_heads, cfg.predictor_hidden_dim) for _ in range(cfg.predictor_num_layers)]
        )
        self.net = nn.Sequential(
            nn.LayerNorm(cfg.model_dim),
            nn.Linear(cfg.model_dim, cfg.predictor_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.predictor_hidden_dim, cfg.model_dim),
        )

    def features(self, x: Tensor) -> Tensor:
        mask = torch.ones((x.size(1), x.size(1)), device=x.device, dtype=torch.bool).triu(1)
        for block in self.blocks:
            x = block(x, mask)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self.net(self.features(x))


def offdiag_ms(cov: Tensor) -> Tensor:
    d = cov.size(0)
    return cov.flatten()[:-1].view(d - 1, d + 1)[:, 1:].square().mean()


class PureByteJEPA(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        self.online_encoder = SequenceEncoder(cfg, causal=True)
        self.target_encoder = SequenceEncoder(cfg, causal=False)
        self.target_encoder.load_state_dict(self.online_encoder.state_dict(), strict=True)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(cfg)
        self.lm_head = nn.Sequential(
            nn.LayerNorm(2 * cfg.model_dim),
            nn.Linear(2 * cfg.model_dim, cfg.lm_probe_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.lm_probe_hidden_dim, cfg.vocab_size),
        )

    def _split(self, byte_ids: Tensor) -> tuple[Tensor, Tensor]:
        return byte_ids[:, :-1], byte_ids[:, 1:]

    def _payload_mask(self, ids: Tensor) -> Tensor:
        return (ids >= self.cfg.byte_offset) & (ids < self.cfg.byte_offset + self.cfg.byte_count)

    def _delta_targets(self, states: Tensor) -> Tensor:
        prev = states[:, :-1].float()
        nxt = states[:, 1:].float()
        basis = F.normalize(prev, dim=-1, eps=1e-8)
        parallel = (nxt * basis).sum(dim=-1, keepdim=True)
        return nxt - parallel * basis

    def _predict(self, byte_ids: Tensor, *, need_target: bool) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
        ctx_ids, tgt_ids = self._split(byte_ids)
        ctx = self.online_encoder(ctx_ids)
        pred = self.predictor(ctx)
        target = None
        if need_target:
            with torch.no_grad():
                target = self._delta_targets(self.target_encoder(byte_ids))
        return tgt_ids, self._payload_mask(tgt_ids), ctx.float(), pred.float(), target

    def _lm_nll_sum(self, ctx: Tensor, pred: Tensor, targets: Tensor, mask: Tensor, *, detach: bool) -> Tensor:
        features = torch.cat([ctx, pred], dim=-1).reshape(-1, 2 * self.cfg.model_dim)
        if detach:
            features = features.detach()
        nll = F.cross_entropy(self.lm_head(features), targets.reshape(-1), reduction="none")
        return nll[mask.reshape(-1)].sum()

    def _repr_reg(self, pred: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        flat = pred[mask].float()
        if flat.size(0) <= 1:
            z = pred.new_zeros(())
            return z, z, z
        var_loss = F.relu(1.0 - torch.sqrt(flat.var(dim=0, unbiased=False) + 1e-4)).mean()
        centered = flat - flat.mean(dim=0, keepdim=True)
        cov_loss = offdiag_ms(centered.T @ centered / max(flat.size(0) - 1, 1))
        return self.cfg.var_reg_weight * var_loss + self.cfg.cov_reg_weight * cov_loss, var_loss, cov_loss

    def forward(self, byte_ids: Tensor) -> dict[str, Tensor]:
        targets, mask, ctx, pred, target = self._predict(byte_ids, need_target=True)
        assert target is not None
        num = mask.float().sum().clamp_min(1.0)
        mse = (pred - target).square().mean(dim=-1)
        base_loss = mse[mask].sum() / num
        reg_loss, var_loss, cov_loss = self._repr_reg(pred, mask)
        lm_loss = self._lm_nll_sum(ctx, pred, targets, mask, detach=True) / num
        loss = base_loss + reg_loss + self.cfg.lm_probe_weight * lm_loss
        return {
            "loss": loss,
            "base_loss": base_loss.detach(),
            "lm_probe_loss": lm_loss.detach(),
            "repr_reg_loss": reg_loss.detach(),
            "var_loss": var_loss.detach(),
            "cov_loss": cov_loss.detach(),
            "num_bytes": num.detach(),
        }

    def validation_step(self, byte_ids: Tensor) -> dict[str, Tensor]:
        targets, mask, ctx, pred, _ = self._predict(byte_ids, need_target=False)
        num = mask.float().sum().clamp_min(1.0)
        nll = self._lm_nll_sum(ctx, pred, targets, mask, detach=False)
        return {"loss_sum": nll.detach(), "num_bytes": num.detach(), "nll_sum_nat": nll.detach()}

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        d = self.cfg.target_ema_decay
        for tgt, src in zip(self.target_encoder.parameters(), self.online_encoder.parameters(), strict=True):
            tgt.data.mul_(d).add_(src.data, alpha=1.0 - d)

    def export_state_dict(self) -> dict[str, Tensor]:
        return self.state_dict()


def load_data_shard(file: Path, cfg: Cfg) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    n = int(header[2])
    vals = np.fromfile(file, dtype="<u2", count=n, offset=header_bytes)
    if header.size != 256 or int(header[0]) != 20240520 or vals.size != n:
        raise ValueError(f"bad shard: {file}")
    lo, hi = (int(vals.min()), int(vals.max())) if vals.size else (0, 0)
    t = torch.from_numpy(vals.astype(np.uint16, copy=False))
    if lo < 0 or hi >= cfg.vocab_size:
        raise ValueError(f"token id out of range in {file}: {lo}..{hi}")
    return t


class ByteStream:
    def __init__(self, pattern: str, cfg: Cfg):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        self.cfg, self.i, self.pos = cfg, 0, 0
        self.vals = load_data_shard(self.files[0], cfg)

    def take(self, n: int) -> Tensor:
        chunks = []
        while n:
            if self.pos >= self.vals.numel():
                self.i = (self.i + 1) % len(self.files)
                self.vals = load_data_shard(self.files[self.i], self.cfg)
                self.pos = 0
            k = min(n, self.vals.numel() - self.pos)
            chunks.append(self.vals[self.pos : self.pos + k])
            self.pos += k
            n -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


def next_batch(stream: ByteStream, cfg: Cfg, device: torch.device, rank: int, world_size: int) -> Tensor:
    denom = world_size * cfg.grad_accum_steps
    if cfg.train_batch_bytes % denom != 0:
        raise ValueError(f"TRAIN_BATCH_BYTES={cfg.train_batch_bytes} must divide WORLD_SIZE*GRAD_ACCUM_STEPS={denom}")
    local = cfg.train_batch_bytes // denom
    if local % cfg.train_seq_len != 0:
        raise ValueError(f"per-rank microbatch bytes={local} must divide TRAIN_SEQ_LEN={cfg.train_seq_len}")
    chunk = stream.take(local * world_size)
    start = rank * local
    return chunk[start : start + local].to(torch.int64).reshape(-1, cfg.train_seq_len).to(device, non_blocking=True)


def load_validation(pattern: str, cfg: Cfg) -> tuple[Tensor, int, bool]:
    vals = torch.cat([load_data_shard(Path(p), cfg) for p in sorted(glob.glob(pattern))]).contiguous()
    usable = ((vals.numel() - 1) // cfg.train_seq_len) * cfg.train_seq_len
    vals = vals[: usable + 1]
    full = vals.numel() - 1
    capped = cfg.val_max_bytes > 0 and full > cfg.val_max_bytes
    if capped:
        keep = (cfg.val_max_bytes // cfg.train_seq_len) * cfg.train_seq_len
        vals = vals[: keep + 1].contiguous()
    return vals, full, capped


def eval_validation(
    cfg: Cfg,
    model: PureByteJEPA,
    device: torch.device,
    val_bytes: Tensor,
    full: bool,
    rank: int,
    world_size: int,
) -> dict[str, float | None]:
    batch_seqs = max((cfg.val_batch_size // max(world_size, 1)) // cfg.train_seq_len, 1)
    total_seqs = (val_bytes.numel() - 1) // cfg.train_seq_len
    per_rank = (total_seqs + world_size - 1) // world_size
    rank_start = min(rank * per_rank, total_seqs)
    rank_end = min(rank_start + per_rank, total_seqs)
    nll_sum = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for s in range(rank_start, rank_end, batch_seqs):
            e = min(s + batch_seqs, rank_end)
            local = val_bytes[s * cfg.train_seq_len : e * cfg.train_seq_len + 1].to(device, torch.int64, non_blocking=True)
            batch = local.unfold(0, cfg.train_seq_len + 1, cfg.train_seq_len).contiguous()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                m = model.validation_step(batch)
            nll_sum += m["nll_sum_nat"].to(torch.float64)
            byte_count += m["num_bytes"].to(torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(nll_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    bpb = float((nll_sum / (math.log(2.0) * byte_count)).item())
    model.train()
    return {"proxy_loss": float((nll_sum / byte_count).item()), "subset_val_bpb": bpb, "val_bpb": bpb if full else None}


DROP_PATTERNS = ("target_encoder",)
KEEP_FP16_PATTERNS = (
    "lm_head",
    "online_encoder.tok_emb",
    "online_encoder.pos_emb",
    "predictor.net",
    "predictor.blocks.0.attn",
    "online_encoder.blocks.0.attn",
    "online_encoder.blocks.1.attn",
    "online_encoder.blocks.0.mlp",
    "online_encoder.blocks.1.mlp",
    "predictor.blocks.0.mlp",
)
KEEP_FP32_PATTERNS = ("lag_weights",)


def q_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    x = t.float()
    if x.ndim == 2:
        clip = torch.quantile(x.abs(), 0.9999984, dim=1)
        scale = (clip / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(torch.clamp(x, -clip[:, None], clip[:, None]) / scale[:, None]), -127, 127).to(torch.int8)
        return q.contiguous(), scale.to(torch.float16).contiguous()
    clip = float(torch.quantile(x.abs().flatten(), 0.9999984).item()) if x.numel() else 0.0
    scale = torch.tensor(clip / 127.0 if clip > 0 else 1.0)
    q = torch.clamp(torch.round(torch.clamp(x, -clip, clip) / scale), -127, 127).to(torch.int8)
    return q.contiguous(), scale


def quantize_state_dict(state: dict[str, Tensor]) -> tuple[dict[str, Any], dict[str, int]]:
    q, scales, dtypes, passthrough, stats = {}, {}, {}, {}, {"baseline_tensor_bytes": 0, "int8_payload_bytes": 0}
    for name, t in state.items():
        t = t.detach().cpu().contiguous()
        stats["baseline_tensor_bytes"] += t.numel() * t.element_size()
        if any(p in name for p in DROP_PATTERNS):
            continue
        if not torch.is_floating_point(t):
            passthrough[name] = t
            continue
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        if any(p in name for p in KEEP_FP32_PATTERNS):
            passthrough[name] = t.float()
        elif any(p in name for p in KEEP_FP16_PATTERNS):
            passthrough[name] = t.to(torch.float16)
        else:
            q[name], scales[name] = q_tensor(t)
            stats["int8_payload_bytes"] += q[name].numel() + scales[name].numel() * scales[name].element_size()
    return {"q": q, "scales": scales, "dtypes": dtypes, "passthrough": passthrough}, stats


def dequantize_state_dict(obj: dict[str, Any]) -> dict[str, Tensor]:
    out = {}
    for name, q in obj["q"].items():
        s = obj["scales"][name]
        dtype = getattr(torch, obj["dtypes"][name])
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(dtype)
    for name, t in obj["passthrough"].items():
        dtype_name = obj["dtypes"].get(name)
        out[name] = t.to(getattr(torch, dtype_name)).contiguous() if dtype_name else t.contiguous()
    return out


def configure_optimizers(model: PureByteJEPA, cfg: Cfg) -> list[torch.optim.Optimizer]:
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    embed = [p for n, p in named if n.endswith("tok_emb.weight")]
    embed_ids = {id(p) for p in embed}
    control_ids = {id(p) for n, p in named if any(k in n for k in KEEP_FP32_PATTERNS)}
    matrices = [p for _, p in named if p.ndim == 2 and id(p) not in embed_ids and id(p) not in control_ids]
    matrix_ids = {id(p) for p in matrices}
    scalars = [p for _, p in named if id(p) not in embed_ids and id(p) not in matrix_ids]
    opts: list[torch.optim.Optimizer] = []
    adam_kwargs = dict(betas=(cfg.beta1, cfg.beta2), eps=cfg.adam_eps, weight_decay=cfg.weight_decay)
    if torch.cuda.is_available():
        adam_kwargs["fused"] = True
    if embed:
        opts.append(torch.optim.AdamW([{"params": embed, "lr": cfg.embed_lr, "base_lr": cfg.embed_lr}], **adam_kwargs))
    if matrices:
        muon = Muon(matrices, lr=cfg.matrix_lr, momentum=cfg.muon_momentum, backend_steps=cfg.muon_backend_steps)
        muon.param_groups[0]["base_lr"] = cfg.matrix_lr
        opts.append(muon)
    if scalars:
        opts.append(torch.optim.AdamW([{"params": scalars, "lr": cfg.scalar_lr, "base_lr": cfg.scalar_lr}], **adam_kwargs))
    return opts


def lr_scale(cfg: Cfg, step: int, elapsed_ms: float) -> float:
    if cfg.warmdown_frac > 0 and cfg.max_wallclock_seconds > 0:
        frac = min(elapsed_ms / max(1000.0 * cfg.max_wallclock_seconds, 1.0), 1.0)
        start = max(1.0 - cfg.warmdown_frac, 0.0)
        if frac >= start:
            return max((1.0 - frac) / max(cfg.warmdown_frac, 1e-9), cfg.min_lr_scale)
        return 1.0
    avg = elapsed_ms / max(step, 1)
    warmdown_ms = cfg.warmdown_iters * avg
    remaining = max(1000.0 * cfg.max_wallclock_seconds - elapsed_ms, 0.0)
    return max(remaining / warmdown_ms if remaining <= warmdown_ms else 1.0, cfg.min_lr_scale)


def load_tokenizer_check(cfg: Cfg) -> None:
    payload = json.loads(Path(cfg.tokenizer_path).read_text())
    conf = payload["config"]
    assert payload["tokenizer_type"] == "pure_byte"
    assert int(payload["vocab_size"]) == cfg.vocab_size
    assert int(conf["bos_id"]) == cfg.bos_token_id
    assert int(conf["byte_offset"]) == cfg.byte_offset


def main() -> None:
    cfg = Cfg()
    if os.environ.get("DRY_RUN") == "1":
        print(f"dry-run {cfg.run_id}: compact final trainer accepted")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    master = rank == 0
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if cfg.grad_accum_steps <= 0:
        if 8 % world_size != 0:
            raise ValueError(f"WORLD_SIZE={world_size} must divide 8 when GRAD_ACCUM_STEPS is auto")
        cfg.grad_accum_steps = max(1, 8 // world_size)
    load_tokenizer_check(cfg)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()

    run_dir = SCRIPT_DIR / "runs" / cfg.run_id
    skip = torch.tensor(int((run_dir / "summary.json").exists()), device=device)
    if distributed:
        dist.all_reduce(skip, op=dist.ReduceOp.MAX)
    if bool(skip.item()):
        if master:
            print(f"skip {cfg.run_id}: summary exists")
        if distributed:
            dist.destroy_process_group()
        return
    if master:
        run_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    log_path = run_dir / "train.log"

    def log(msg: str, console: bool = True) -> None:
        if not master:
            return
        if console:
            print(msg)
        with log_path.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    code = Path(__file__).read_text()
    if master:
        (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True) + "\n")
        (run_dir / "train_jepa_snapshot.py").write_text(code)
    log(code, console=False)
    log("=" * 100, console=False)

    val_bytes, full_val_bytes, capped = load_validation(cfg.val_files, cfg)
    log(f"dataset:{Path(cfg.data_path).name} train_shards:{len(glob.glob(cfg.train_files))}")
    log(f"val_target_bytes:{val_bytes.numel() - 1} full_target_bytes:{full_val_bytes} subset_only:{int(capped)}")

    base_model = PureByteJEPA(cfg).to(device)
    model: nn.Module = DDP(base_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else base_model
    opts = configure_optimizers(base_model, cfg)
    train_stream = ByteStream(cfg.train_files, cfg)
    log(f"model_params:{sum(p.numel() for p in base_model.parameters())}")
    log(f"world_size:{world_size} grad_accum_steps:{cfg.grad_accum_steps}")
    log(f"iterations:{cfg.iterations} batch_bytes:{cfg.train_batch_bytes} seq_len:{cfg.train_seq_len}")
    log(f"optimizer embed_lr:{cfg.embed_lr:.6g} matrix_lr:{cfg.matrix_lr:.6g} scalar_lr:{cfg.scalar_lr:.6g} beta2:{cfg.beta2:.6g}")
    log(f"schedule warmdown_frac:{cfg.warmdown_frac:.4g} min_lr_scale:{cfg.min_lr_scale:.4g} max_wallclock_seconds:{cfg.max_wallclock_seconds:.1f}")

    def zero_grad() -> None:
        for opt in opts:
            opt.zero_grad(set_to_none=True)

    def all_train_params() -> list[nn.Parameter]:
        out, seen = [], set()
        for opt in opts:
            for group in opt.param_groups:
                for p in group["params"]:
                    if id(p) not in seen:
                        out.append(p)
                        seen.add(id(p))
        return out

    train_params = all_train_params()
    latest_val: dict[str, float | None] | None = None
    train_ms = 0.0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    while step <= cfg.iterations:
        if (cfg.val_loss_every > 0 and step % cfg.val_loss_every == 0) or step == cfg.iterations:
            torch.cuda.synchronize()
            train_ms += 1000.0 * (time.perf_counter() - t0)
            latest_val = eval_validation(cfg, base_model, device, val_bytes, full=not capped, rank=rank, world_size=world_size)
            msg_bpb = f"val_bpb:{latest_val['val_bpb']:.8f}" if latest_val["val_bpb"] is not None else f"val_bpb_subset:{latest_val['subset_val_bpb']:.8f} val_bpb:NA"
            log(f"step:{step}/{cfg.iterations} val_proxy_loss:{latest_val['proxy_loss']:.6f} {msg_bpb} train_time:{train_ms:.0f}ms step_avg:{train_ms / max(step, 1):.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if step == cfg.iterations:
            break

        elapsed = train_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_scale(cfg, step, elapsed)
        frac = min(step / cfg.muon_momentum_warmup_steps, 1.0)
        momentum = (1 - frac) * cfg.muon_momentum_warmup_start + frac * cfg.muon_momentum
        for opt in opts:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
                if "momentum" in group:
                    group["momentum"] = momentum

        zero_grad()
        sums: dict[str, float] = {}
        for micro_step in range(cfg.grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == cfg.grad_accum_steps - 1
            batch = next_batch(train_stream, cfg, device, rank, world_size)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                metrics = model(batch)
            (metrics["loss"] / cfg.grad_accum_steps).backward()
            for k, v in metrics.items():
                if k != "loss":
                    sums[k] = sums.get(k, 0.0) + float(v.detach().item()) / cfg.grad_accum_steps
            sums["loss"] = sums.get("loss", 0.0) + float(metrics["loss"].detach().item()) / cfg.grad_accum_steps
        if cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(train_params, cfg.grad_clip_norm)
        for opt in opts:
            opt.step()
        base_model.update_target_encoder()
        zero_grad()
        if distributed:
            model.require_backward_grad_sync = True
        step += 1

        if step <= 10 or step % cfg.train_log_every == 0:
            torch.cuda.synchronize()
            approx = train_ms + 1000.0 * (time.perf_counter() - t0)
            parts = " ".join(f"{k}:{v:.6f}" for k, v in sums.items())
            log(f"step:{step}/{cfg.iterations} {parts} train_time:{approx:.0f}ms step_avg:{approx / step:.2f}ms")
        reached_cap = train_ms + 1000.0 * (time.perf_counter() - t0) >= 1000.0 * cfg.max_wallclock_seconds
        if distributed:
            reached = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached.item())
        if reached_cap:
            log(f"stopping_early step:{step}/{cfg.iterations}")
            break

    if latest_val is None:
        latest_val = eval_validation(cfg, base_model, device, val_bytes, full=not capped, rank=rank, world_size=world_size)
    raw_path = run_dir / "final_model.pt"
    q_path = run_dir / "final_model.int8.ptz"
    if master:
        export_state = {k: v.detach().cpu().contiguous().clone() for k, v in base_model.export_state_dict().items()}
        torch.save(export_state, raw_path)
        quant_obj, qstats = quantize_state_dict(export_state)
        buf = io.BytesIO()
        torch.save(quant_obj, buf)
        q_path.write_bytes(zlib.compress(buf.getvalue(), level=9))
        log(f"Serialized model:{raw_path.stat().st_size} bytes")
        log(f"Serialized model int8+zlib:{q_path.stat().st_size} bytes")
        log(f"Code size:{len(code.encode())} bytes total_submission_int8_zlib:{q_path.stat().st_size + len(code.encode())} bytes")
    else:
        qstats = {}
    if distributed:
        dist.barrier()

    disk_obj = torch.load(io.BytesIO(zlib.decompress(q_path.read_bytes())), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict(disk_obj), strict=False)
    q_metrics = eval_validation(cfg, base_model, device, val_bytes, full=not capped, rank=rank, world_size=world_size)
    msg_bpb = f"val_bpb:{q_metrics['val_bpb']:.8f}" if q_metrics["val_bpb"] is not None else f"val_bpb_subset:{q_metrics['subset_val_bpb']:.8f} val_bpb:NA"
    log(f"final_int8_zlib_roundtrip val_proxy_loss:{q_metrics['proxy_loss']:.6f} {msg_bpb}")
    if master:
        summary = {
            "run_id": cfg.run_id,
            "raw_final_proxy_val_loss": latest_val["proxy_loss"],
            "raw_final_subset_val_bpb": latest_val["subset_val_bpb"],
            "raw_final_val_bpb": latest_val["val_bpb"],
            "proxy_val_loss": q_metrics["proxy_loss"],
            "subset_val_bpb": q_metrics["subset_val_bpb"],
            "val_bpb": q_metrics["val_bpb"],
            "proxy_only": q_metrics["val_bpb"] is None,
            "validation_bytes": int(val_bytes.numel() - 1),
            "validation_full_bytes": int(full_val_bytes),
            "artifact_model_bytes_int8_zlib": q_path.stat().st_size,
            "artifact_code_bytes": len(code.encode()),
            "artifact_total_bytes_int8_zlib": q_path.stat().st_size + len(code.encode()),
            "world_size": world_size,
            "global_train_batch_bytes": cfg.train_batch_bytes,
            "per_rank_microbatch_bytes": cfg.train_batch_bytes // (world_size * cfg.grad_accum_steps),
            "quant_stats": qstats,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
