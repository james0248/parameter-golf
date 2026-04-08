from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from baseline_gpt import Muon


class EncoderBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, mlp_hidden_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, model_dim),
        )

    def forward(self, x: Tensor, attn_mask: Tensor | None) -> Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SequenceEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_hidden_dim: int,
        max_seq_len: int,
        dropout: float,
        *,
        causal: bool,
    ):
        super().__init__()
        self.causal = causal
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, model_dim))
        self.blocks = nn.ModuleList(
            [EncoderBlock(model_dim, num_heads, mlp_hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(model_dim)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.01)

    def forward(self, token_ids: Tensor) -> Tensor:
        seq_len = token_ids.size(1)
        x = self.tok_emb(token_ids) + self.pos_emb[:, :seq_len, :]
        attn_mask = None
        if self.causal:
            attn_mask = torch.ones((seq_len, seq_len), device=token_ids.device, dtype=torch.bool).triu(diagonal=1)
        for block in self.blocks:
            x = block(x, attn_mask)
        return self.norm(x)


class Predictor(nn.Module):
    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        *,
        num_layers: int = 0,
        num_heads: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        output_dim = model_dim if output_dim is None else output_dim
        self.blocks = nn.ModuleList(
            [EncoderBlock(model_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.net = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.blocks:
            seq_len = x.size(1)
            attn_mask = torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool).triu(diagonal=1)
            for block in self.blocks:
                x = block(x, attn_mask)
        return self.net(x)


def _sylvester_hadamard(order: int) -> Tensor:
    if order <= 0 or order & (order - 1):
        raise ValueError(f"Hadamard order must be a positive power of two, got {order}")
    h = torch.ones((1, 1), dtype=torch.float32)
    while h.size(0) < order:
        top = torch.cat((h, h), dim=1)
        bottom = torch.cat((h, -h), dim=1)
        h = torch.cat((top, bottom), dim=0)
    return h


def _build_fixed_codebook(
    vocab_size: int,
    byte_offset: int,
    byte_count: int,
    model_dim: int,
    kind: str,
    seed: int,
) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    codebook = torch.randn((vocab_size, model_dim), generator=generator, dtype=torch.float32)

    if kind == "fixed_random":
        return F.normalize(codebook, dim=-1, eps=1e-8)

    if kind == "fixed_hadamard":
        if model_dim < byte_count:
            raise ValueError(
                f"TARGET_CODEBOOK='fixed_hadamard' requires MODEL_DIM >= BYTE_COUNT; got {model_dim} < {byte_count}"
            )
        payload_codes = _sylvester_hadamard(byte_count)
        if model_dim > byte_count:
            payload_codes = F.pad(payload_codes, (0, model_dim - byte_count))
        codebook[byte_offset : byte_offset + byte_count] = payload_codes
        return F.normalize(codebook, dim=-1, eps=1e-8)

    raise ValueError(f"Unknown TARGET_CODEBOOK={kind!r}")


def _off_diagonal_mean_square(cov: Tensor) -> Tensor:
    dim = cov.size(0)
    if dim <= 1:
        return cov.new_zeros(())
    return cov.flatten()[:-1].view(dim - 1, dim + 1)[:, 1:].square().mean()


def _orthogonal_residual(source: Tensor, anchor: Tensor) -> Tensor:
    anchor_basis = F.normalize(anchor.float(), dim=-1, eps=1e-8)
    source_float = source.float()
    return source_float - (source_float * anchor_basis).sum(dim=-1, keepdim=True) * anchor_basis


def _any_orthogonal_unit(anchor: Tensor) -> Tensor:
    anchor_basis = F.normalize(anchor.float(), dim=-1, eps=1e-8)
    idx = anchor_basis.abs().argmin(dim=-1, keepdim=True)
    basis = torch.zeros_like(anchor_basis).scatter_(-1, idx, 1.0)
    ortho = basis - (basis * anchor_basis).sum(dim=-1, keepdim=True) * anchor_basis
    return F.normalize(ortho, dim=-1, eps=1e-8)


def _rotation_scale_delta(source: Tensor, anchor: Tensor) -> Tensor:
    anchor_float = anchor.float()
    source_float = source.float()
    anchor_norm = anchor_float.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    source_norm = source_float.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    anchor_dir = anchor_float / anchor_norm
    source_dir = source_float / source_norm
    cos_sim = (anchor_dir * source_dir).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    tangent = source_dir - cos_sim * anchor_dir
    sin_norm = tangent.norm(dim=-1, keepdim=True)
    tangent_dir = tangent / sin_norm.clamp_min(1e-8)
    antipodal = _any_orthogonal_unit(anchor_dir)
    near_parallel = sin_norm < 1e-6
    tangent_dir = torch.where((near_parallel & (cos_sim < 0.0)).expand_as(tangent_dir), antipodal, tangent_dir)
    theta = torch.atan2(sin_norm, cos_sim)
    theta = torch.where(near_parallel & (cos_sim < 0.0), torch.full_like(theta, float(torch.pi)), theta)
    theta = torch.where(near_parallel & (cos_sim >= 0.0), torch.zeros_like(theta), theta)
    log_scale = torch.log(source_norm / anchor_norm)
    return theta * tangent_dir + log_scale * anchor_dir


def _reconstruct_rotation_scale_next_state(anchor: Tensor, diff: Tensor) -> Tensor:
    anchor_float = anchor.float()
    diff_float = diff.float()
    anchor_norm = anchor_float.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    anchor_dir = anchor_float / anchor_norm
    log_scale = (diff_float * anchor_dir).sum(dim=-1, keepdim=True)
    delta_rot = diff_float - log_scale * anchor_dir
    theta = delta_rot.norm(dim=-1, keepdim=True)
    tangent_dir = delta_rot / theta.clamp_min(1e-8)
    next_dir = torch.cos(theta) * anchor_dir + torch.sin(theta) * tangent_dir
    next_dir = torch.where((theta < 1e-6).expand_as(next_dir), anchor_dir, next_dir)
    return anchor_norm * torch.exp(log_scale) * next_dir


def _sphere_delta(source: Tensor, anchor: Tensor) -> Tensor:
    anchor_dir = F.normalize(anchor.float(), dim=-1, eps=1e-8)
    source_dir = F.normalize(source.float(), dim=-1, eps=1e-8)
    return _orthogonal_residual(source_dir, anchor_dir)


def _reconstruct_sphere_delta_next_state(anchor: Tensor, diff: Tensor) -> Tensor:
    anchor_dir = F.normalize(anchor.float(), dim=-1, eps=1e-8)
    tangent = _orthogonal_residual(diff.float(), anchor_dir)
    tangent_norm = tangent.norm(dim=-1, keepdim=True)
    max_norm = 1.0 - 1e-6
    safe_scale = torch.where(
        tangent_norm > max_norm,
        torch.full_like(tangent_norm, max_norm) / tangent_norm.clamp_min(1e-8),
        torch.ones_like(tangent_norm),
    )
    tangent = tangent * safe_scale
    radial = torch.sqrt((1.0 - tangent.square().sum(dim=-1, keepdim=True)).clamp_min(1e-8))
    return radial * anchor_dir + tangent


def _reconstruct_additive_delta_next_state(
    anchor: Tensor,
    diff: Tensor,
    *,
    normalize_anchor: bool,
    normalize_output: bool,
) -> Tensor:
    anchor_repr = anchor.float()
    if normalize_anchor:
        anchor_repr = F.normalize(anchor_repr, dim=-1, eps=1e-8)
    next_state = anchor_repr + diff.float()
    if normalize_output:
        next_state = F.normalize(next_state, dim=-1, eps=1e-8)
    return next_state


def _reconstruct_contextual_delta_next_state(anchor: Tensor, diff: Tensor, parallel_shift: Tensor) -> Tensor:
    anchor_float = anchor.float()
    anchor_basis = F.normalize(anchor_float, dim=-1, eps=1e-8)
    anchor_norm = anchor_float.norm(dim=-1, keepdim=True)
    orth_diff = _orthogonal_residual(diff.float(), anchor_float)
    return (anchor_norm + parallel_shift.float()) * anchor_basis + orth_diff


def _scaled_contextual_delta(diff: Tensor, anchor: Tensor, log_norm: Tensor) -> Tensor:
    orth_diff = _orthogonal_residual(diff.float(), anchor.float())
    diff_dir = F.normalize(orth_diff, dim=-1, eps=1e-8)
    return diff_dir * torch.exp(log_norm.float())


class PureByteJEPA(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        byte_offset: int,
        byte_count: int,
        seq_len: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_hidden_dim: int,
        predictor_hidden_dim: int,
        predictor_num_layers: int,
        predictor_num_heads: int,
        predictor_input: str,
        predictor_style: str,
        action_dim: int,
        transition_hidden_dim: int,
        predictor_model_layers: int,
        predictor_model_heads: int,
        predictor_model_mlp_hidden_dim: int,
        dropout: float,
        target_ema_decay: float,
        eval_logit_scale: float,
        jepa_loss: str,
        eval_bridge: str,
        target_codebook: str,
        num_target_codebooks: int,
        contextual_target: bool,
        target_source: str,
        target_encoder_causal: bool,
        target_encoder_mode: str,
        target_stop_grad: bool,
        target_anchor_weight: float,
        prototype_ema_decay: float,
        contextual_residual_scale: float,
        contextual_aux_weight: float,
        lm_probe_weight: float,
        lm_probe_input: str,
        lm_probe_hidden_dim: int,
        target_future_steps: int,
        contextual_delta_normalization: str,
        contextual_delta_reference: str,
        contextual_delta_parallel_weight: float,
        contextual_delta_magnitude_weight: float,
        contextual_delta_teacher_orth_weight: float,
        contextual_delta_contrastive_teacher_anchor: bool,
        rotation_delta_loss_target: str,
        additive_delta_normalization: str,
        contrastive_weight: float,
        contrastive_temperature: float,
        contrastive_samples: int,
        contrastive_future_steps: int,
        contrastive_mode: str,
        var_reg_weight: float,
        cov_reg_weight: float,
        target_var_reg_weight: float,
        target_cov_reg_weight: float,
    ):
        super().__init__()
        if num_target_codebooks <= 0:
            raise ValueError(f"NUM_TARGET_CODEBOOKS must be positive, got {num_target_codebooks}")
        if predictor_input not in {"context_latent", "tokens", "context_plus_tokens"}:
            raise ValueError(f"Unknown PREDICTOR_INPUT={predictor_input!r}")
        if predictor_style not in {"direct", "action_transition"}:
            raise ValueError(f"Unknown PREDICTOR_STYLE={predictor_style!r}")
        if target_source not in {"codebook", "contextual_only", "contextual_delta", "rotation_scale_delta", "sphere_delta", "additive_delta"}:
            raise ValueError(f"Unknown TARGET_SOURCE={target_source!r}")
        if target_encoder_mode not in {"ema", "shared"}:
            raise ValueError(f"Unknown TARGET_ENCODER_MODE={target_encoder_mode!r}")
        if lm_probe_input not in {
            "pred_target",
            "pred_residual",
            "context_plus_pred",
            "context_plus_scaled_pred",
            "context_plus_pred_parallel",
            "context_plus_pred_magnitude",
            "action",
            "context_plus_action",
        }:
            raise ValueError(f"Unknown LM_PROBE_INPUT={lm_probe_input!r}")
        if target_future_steps <= 0:
            raise ValueError(f"TARGET_FUTURE_STEPS must be positive, got {target_future_steps}")
        if contextual_delta_normalization not in {"normalized", "raw"}:
            raise ValueError(f"Unknown CONTEXTUAL_DELTA_NORMALIZATION={contextual_delta_normalization!r}")
        if contextual_delta_reference not in {"anchor", "stepwise"}:
            raise ValueError(f"Unknown CONTEXTUAL_DELTA_REFERENCE={contextual_delta_reference!r}")
        if contextual_delta_parallel_weight < 0:
            raise ValueError(
                f"CONTEXTUAL_DELTA_PARALLEL_WEIGHT must be non-negative, got {contextual_delta_parallel_weight}"
            )
        if contextual_delta_magnitude_weight < 0:
            raise ValueError(
                f"CONTEXTUAL_DELTA_MAGNITUDE_WEIGHT must be non-negative, got {contextual_delta_magnitude_weight}"
            )
        if contextual_delta_teacher_orth_weight < 0:
            raise ValueError(
                f"CONTEXTUAL_DELTA_TEACHER_ORTH_WEIGHT must be non-negative, got {contextual_delta_teacher_orth_weight}"
            )
        if rotation_delta_loss_target not in {"diff", "next_state"}:
            raise ValueError(f"Unknown ROTATION_DELTA_LOSS_TARGET={rotation_delta_loss_target!r}")
        if additive_delta_normalization not in {"none", "states", "diff", "states_and_diff"}:
            raise ValueError(f"Unknown ADDITIVE_DELTA_NORMALIZATION={additive_delta_normalization!r}")
        if contrastive_temperature <= 0:
            raise ValueError(f"CONTRASTIVE_TEMPERATURE must be positive, got {contrastive_temperature}")
        if contrastive_samples < 0:
            raise ValueError(f"CONTRASTIVE_SAMPLES must be non-negative, got {contrastive_samples}")
        if contrastive_future_steps <= 0:
            raise ValueError(f"CONTRASTIVE_FUTURE_STEPS must be positive, got {contrastive_future_steps}")
        if contrastive_mode not in {"delta", "future_state"}:
            raise ValueError(f"Unknown CONTRASTIVE_MODE={contrastive_mode!r}")
        if contextual_target and num_target_codebooks != 1:
            raise ValueError("CONTEXTUAL_TARGET=1 currently requires NUM_TARGET_CODEBOOKS=1")
        if target_source in {"contextual_only", "contextual_delta", "rotation_scale_delta", "sphere_delta", "additive_delta"} and num_target_codebooks != 1:
            raise ValueError("TARGET_SOURCE contextual variants currently require NUM_TARGET_CODEBOOKS=1")
        if target_codebook == "ema" and num_target_codebooks != 1:
            raise ValueError("TARGET_CODEBOOK='ema' only supports NUM_TARGET_CODEBOOKS=1")
        if predictor_style == "action_transition" and action_dim <= 0:
            raise ValueError(f"ACTION_DIM must be positive for action_transition, got {action_dim}")
        if lm_probe_input in {"action", "context_plus_action"} and predictor_style != "action_transition":
            raise ValueError(f"LM_PROBE_INPUT={lm_probe_input!r} requires PREDICTOR_STYLE='action_transition'")
        self.vocab_size = vocab_size
        self.byte_offset = byte_offset
        self.byte_count = byte_count
        self.byte_token_max = byte_offset + byte_count
        self.seq_len = seq_len
        self.target_ema_decay = target_ema_decay
        self.eval_logit_scale = float(eval_logit_scale)
        self.jepa_loss = jepa_loss
        self.eval_bridge = eval_bridge
        self.target_codebook = target_codebook
        self.num_target_codebooks = num_target_codebooks
        self.contextual_target = contextual_target
        self.target_source = target_source
        self.target_encoder_causal = target_encoder_causal
        self.target_encoder_mode = target_encoder_mode
        self.target_stop_grad = bool(target_stop_grad)
        self.target_anchor_weight = float(target_anchor_weight)
        self.prototype_ema_decay = float(prototype_ema_decay)
        self.contextual_residual_scale = float(contextual_residual_scale)
        self.contextual_aux_weight = float(contextual_aux_weight)
        self.lm_probe_weight = float(lm_probe_weight)
        self.lm_probe_input = lm_probe_input
        self.lm_probe_hidden_dim = int(lm_probe_hidden_dim)
        self.target_future_steps = int(target_future_steps)
        self.contextual_delta_normalization = contextual_delta_normalization
        self.contextual_delta_reference = contextual_delta_reference
        self.contextual_delta_parallel_weight = float(contextual_delta_parallel_weight)
        self.contextual_delta_magnitude_weight = float(contextual_delta_magnitude_weight)
        self.contextual_delta_teacher_orth_weight = float(contextual_delta_teacher_orth_weight)
        self.contextual_delta_contrastive_teacher_anchor = bool(contextual_delta_contrastive_teacher_anchor)
        self.rotation_delta_loss_target = rotation_delta_loss_target
        self.additive_delta_normalization = additive_delta_normalization
        self.contrastive_weight = float(contrastive_weight)
        self.contrastive_temperature = float(contrastive_temperature)
        self.contrastive_samples = int(contrastive_samples)
        self.contrastive_future_steps = int(contrastive_future_steps)
        self.contrastive_mode = contrastive_mode
        self.var_reg_weight = float(var_reg_weight)
        self.cov_reg_weight = float(cov_reg_weight)
        self.target_var_reg_weight = float(target_var_reg_weight)
        self.target_cov_reg_weight = float(target_cov_reg_weight)
        self.predictor_input = predictor_input
        self.predictor_style = predictor_style
        self.action_dim = int(action_dim)
        self.transition_hidden_dim = int(transition_hidden_dim)
        self.uses_anchor_codebook = self.target_source == "codebook"

        self.online_encoder = SequenceEncoder(
            vocab_size=vocab_size,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            max_seq_len=seq_len,
            dropout=dropout,
            causal=True,
        )
        self.target_encoder: SequenceEncoder | None = None
        needs_target_encoder = (
            self.contextual_target
            or self.contextual_aux_weight > 0.0
            or self.target_source in {"contextual_only", "contextual_delta", "rotation_scale_delta", "sphere_delta", "additive_delta"}
        )
        if self.target_encoder_mode == "ema" and needs_target_encoder:
            self.target_encoder = SequenceEncoder(
                vocab_size=vocab_size,
                model_dim=model_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                max_seq_len=seq_len,
                dropout=dropout,
                causal=target_encoder_causal,
            )
            self.target_encoder.load_state_dict(self.online_encoder.state_dict(), strict=True)
            for param in self.target_encoder.parameters():
                param.requires_grad_(False)

        self.target_codebooks = nn.Parameter(
            torch.empty(num_target_codebooks, vocab_size, model_dim),
            requires_grad=False,
        )
        self.eval_codebooks = nn.Parameter(
            torch.empty(num_target_codebooks, vocab_size, model_dim),
            requires_grad=False,
        )
        with torch.no_grad():
            if self.uses_anchor_codebook:
                if self.target_codebook == "ema":
                    self.target_codebooks[0].copy_(self.online_encoder.tok_emb.weight)
                else:
                    for codebook_idx in range(self.num_target_codebooks):
                        self.target_codebooks[codebook_idx].copy_(
                            _build_fixed_codebook(
                                vocab_size,
                                byte_offset,
                                byte_count,
                                model_dim,
                                self.target_codebook,
                                seed=codebook_idx,
                            )
                        )
                self.eval_codebooks.copy_(self.target_codebooks)
            else:
                self.target_codebooks.zero_()
                self.eval_codebooks.zero_()

        self.predictor_encoder: SequenceEncoder | None = None
        self.predictor_fusion: nn.Module | None = None
        if self.predictor_input in {"tokens", "context_plus_tokens"}:
            self.predictor_encoder = SequenceEncoder(
                vocab_size=vocab_size,
                model_dim=model_dim,
                num_layers=predictor_model_layers,
                num_heads=predictor_model_heads,
                mlp_hidden_dim=predictor_model_mlp_hidden_dim,
                max_seq_len=seq_len,
                dropout=dropout,
                causal=True,
            )
            if self.predictor_input == "context_plus_tokens":
                self.predictor_fusion = nn.Sequential(
                    nn.LayerNorm(2 * model_dim),
                    nn.Linear(2 * model_dim, model_dim),
                    nn.GELU(),
                    nn.Linear(model_dim, model_dim),
                )
        self.predictor: nn.Module | None = None
        self.action_head: nn.Module | None = None
        self.transition_head: nn.Module | None = None
        self.contextual_delta_parallel_head: nn.Module | None = None
        self.contextual_delta_magnitude_head: nn.Module | None = None
        if self.predictor_style == "action_transition":
            self.action_head = nn.Sequential(
                nn.LayerNorm(model_dim),
                nn.Linear(model_dim, predictor_hidden_dim),
                nn.GELU(),
                nn.Linear(predictor_hidden_dim, self.action_dim),
            )
            self.transition_head = nn.Sequential(
                nn.LayerNorm(model_dim + self.action_dim),
                nn.Linear(model_dim + self.action_dim, self.transition_hidden_dim),
                nn.GELU(),
                nn.Linear(self.transition_hidden_dim, model_dim),
            )
        else:
            self.predictor = Predictor(
                model_dim=model_dim,
                hidden_dim=predictor_hidden_dim,
                output_dim=model_dim * self.num_target_codebooks,
                num_layers=predictor_num_layers,
                num_heads=predictor_num_heads,
                dropout=dropout,
            )
            if self.target_source == "contextual_delta" and self.contextual_delta_parallel_weight > 0.0:
                self.contextual_delta_parallel_head = nn.Sequential(
                    nn.LayerNorm(model_dim),
                    nn.Linear(model_dim, predictor_hidden_dim),
                    nn.GELU(),
                    nn.Linear(predictor_hidden_dim, self.num_target_codebooks),
                )
            if self.target_source == "contextual_delta" and self.contextual_delta_magnitude_weight > 0.0:
                self.contextual_delta_magnitude_head = nn.Sequential(
                    nn.LayerNorm(model_dim),
                    nn.Linear(model_dim, predictor_hidden_dim),
                    nn.GELU(),
                    nn.Linear(predictor_hidden_dim, self.num_target_codebooks),
                )
        self.contextual_predictor: Predictor | None = None
        if self.contextual_aux_weight > 0.0:
            self.contextual_predictor = Predictor(
                model_dim=model_dim,
                hidden_dim=predictor_hidden_dim,
                num_layers=predictor_num_layers,
                num_heads=predictor_num_heads,
                dropout=dropout,
            )

        self.lm_head: nn.Module | None = None
        if self.lm_probe_weight > 0.0:
            if self.lm_probe_input in {"pred_target", "pred_residual"}:
                lm_probe_feature_dim = model_dim * self.num_target_codebooks
            elif self.lm_probe_input in {"context_plus_pred", "context_plus_scaled_pred"}:
                lm_probe_feature_dim = 2 * model_dim * self.num_target_codebooks
            elif self.lm_probe_input == "context_plus_pred_parallel":
                lm_probe_feature_dim = (2 * model_dim + 1) * self.num_target_codebooks
            elif self.lm_probe_input == "context_plus_pred_magnitude":
                lm_probe_feature_dim = (2 * model_dim + 1) * self.num_target_codebooks
            elif self.lm_probe_input == "action":
                lm_probe_feature_dim = self.action_dim
            else:
                lm_probe_feature_dim = model_dim + self.action_dim
            lm_layers: list[nn.Module] = [nn.LayerNorm(lm_probe_feature_dim)]
            if self.lm_probe_hidden_dim > 0:
                lm_layers.extend([
                    nn.Linear(lm_probe_feature_dim, self.lm_probe_hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.lm_probe_hidden_dim, vocab_size),
                ])
            else:
                lm_layers.append(nn.Linear(lm_probe_feature_dim, vocab_size))
            self.lm_head = nn.Sequential(*lm_layers)

    def forward(self, byte_ids: Tensor, return_validation: bool = False) -> dict[str, Tensor]:
        if return_validation:
            return self.validation_step(byte_ids)
        return self.training_step(byte_ids)

    def _split_inputs_and_targets(self, byte_ids: Tensor) -> tuple[Tensor, Tensor]:
        if byte_ids.ndim != 2:
            raise ValueError(f"Expected [batch, seq] byte ids, got {tuple(byte_ids.shape)}")
        if byte_ids.size(1) < 2:
            raise ValueError("Sequence length must be at least 2 for next-byte JEPA")
        return byte_ids[:, :-1], byte_ids[:, 1:]

    def _payload_mask(self, target_ids: Tensor) -> Tensor:
        return (target_ids >= self.byte_offset) & (target_ids < self.byte_token_max)

    def _anchor_targets(self, target_ids: Tensor) -> Tensor:
        if not self.uses_anchor_codebook:
            raise RuntimeError("Anchor targets requested for a contextual-only target source")
        with torch.no_grad():
            return torch.stack(
                [F.embedding(target_ids, self.target_codebooks[i]) for i in range(self.num_target_codebooks)],
                dim=2,
            )

    def _contextual_residual_targets(self, anchor_targets: Tensor, contextual_targets: Tensor) -> Tensor:
        anchor_repr = F.normalize(anchor_targets.float(), dim=-1, eps=1e-8)
        residual = _orthogonal_residual(contextual_targets, anchor_targets)
        residual_dir = residual / residual.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        target_repr = F.normalize(anchor_repr + self.contextual_residual_scale * residual_dir, dim=-1, eps=1e-8)
        return target_repr.to(dtype=anchor_targets.dtype)

    def _contextual_delta_targets(self, contextual_states: Tensor, *, steps: int = 1) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.contextual_delta_reference == "stepwise" and steps > 1:
            prev_targets = contextual_states[:, steps - 1 : -1, :].unsqueeze(2)
        else:
            prev_targets = contextual_states[:, :-steps, :].unsqueeze(2)
        next_targets = contextual_states[:, steps:, :].unsqueeze(2)
        prev_basis = F.normalize(prev_targets.float(), dim=-1, eps=1e-8)
        parallel_coeff = (next_targets.float() * prev_basis).sum(dim=-1, keepdim=True)
        prev_norm = prev_targets.float().norm(dim=-1, keepdim=True)
        delta_targets = next_targets.float() - parallel_coeff * prev_basis
        delta_norm = delta_targets.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        parallel_shift = parallel_coeff - prev_norm
        delta_log_norm = torch.log(delta_norm)
        if self.contextual_delta_normalization == "normalized":
            delta_targets = delta_targets / delta_norm
        return (
            next_targets,
            delta_targets.to(dtype=next_targets.dtype),
            parallel_shift.to(dtype=next_targets.dtype),
            delta_log_norm.to(dtype=next_targets.dtype),
        )

    def _rotation_scale_delta_targets(self, contextual_states: Tensor, *, steps: int = 1) -> tuple[Tensor, Tensor]:
        prev_targets = contextual_states[:, :-steps, :].unsqueeze(2)
        next_targets = contextual_states[:, steps:, :].unsqueeze(2)
        delta_targets = _rotation_scale_delta(next_targets, prev_targets)
        return next_targets, delta_targets.to(dtype=next_targets.dtype)

    def _sphere_delta_targets(self, contextual_states: Tensor, *, steps: int = 1) -> tuple[Tensor, Tensor]:
        prev_targets = contextual_states[:, :-steps, :].unsqueeze(2)
        next_targets = contextual_states[:, steps:, :].unsqueeze(2)
        prev_unit = F.normalize(prev_targets.float(), dim=-1, eps=1e-8)
        next_unit = F.normalize(next_targets.float(), dim=-1, eps=1e-8)
        delta_targets = _sphere_delta(next_unit, prev_unit)
        return next_unit.to(dtype=next_targets.dtype), delta_targets.to(dtype=next_targets.dtype)

    def _additive_delta_targets(self, contextual_states: Tensor, *, steps: int = 1) -> tuple[Tensor, Tensor]:
        prev_targets = contextual_states[:, :-steps, :].unsqueeze(2).float()
        next_targets = contextual_states[:, steps:, :].unsqueeze(2).float()
        if self.additive_delta_normalization in {"states", "states_and_diff"}:
            prev_repr = F.normalize(prev_targets, dim=-1, eps=1e-8)
            next_repr = F.normalize(next_targets, dim=-1, eps=1e-8)
        else:
            prev_repr = prev_targets
            next_repr = next_targets
        delta_targets = next_repr - prev_repr
        if self.additive_delta_normalization in {"diff", "states_and_diff"}:
            delta_targets = F.normalize(delta_targets, dim=-1, eps=1e-8)
        return next_repr.to(dtype=next_targets.dtype), delta_targets.to(dtype=next_targets.dtype)

    def _encode_target_states(self, byte_ids: Tensor) -> Tensor:
        encoder: SequenceEncoder | None
        if self.target_encoder_mode == "shared":
            encoder = self.online_encoder
        else:
            encoder = self.target_encoder
        if encoder is None:
            raise RuntimeError("Contextual target source requested without an available target encoder")
        if self.target_stop_grad:
            with torch.no_grad():
                return encoder(byte_ids)
        return encoder(byte_ids)

    def _contrastive_infonce_loss(
        self,
        pred_target: Tensor,
        pred_parallel: Tensor | None,
        pred_magnitude: Tensor | None,
        context_latents: Tensor,
        teacher_states: Tensor | None,
        payload_mask: Tensor,
    ) -> Tensor:
        if self.contrastive_weight <= 0.0 or self.contrastive_samples == 0 or teacher_states is None:
            return pred_target.new_zeros(())
        if teacher_states.size(1) <= self.contrastive_future_steps:
            return pred_target.new_zeros(())
        if self.contrastive_mode == "future_state":
            key = teacher_states[:, self.contrastive_future_steps :, :].unsqueeze(2).float()
            context_anchor = context_latents[:, : key.size(1)].unsqueeze(2).float()
            teacher_anchor = teacher_states[:, : key.size(1), :].unsqueeze(2).float()
            delta_anchor = teacher_anchor if self.contextual_delta_contrastive_teacher_anchor else context_anchor
            if self.target_source == "contextual_only":
                query = pred_target[:, : key.size(1)].float()
            elif self.target_source == "rotation_scale_delta":
                query = _reconstruct_rotation_scale_next_state(
                    context_anchor,
                    pred_target[:, : key.size(1)],
                ).float()
            elif self.target_source == "sphere_delta":
                query = _reconstruct_sphere_delta_next_state(
                    context_anchor,
                    pred_target[:, : key.size(1)],
                ).float()
            elif self.target_source == "additive_delta":
                query = _reconstruct_additive_delta_next_state(
                    context_anchor,
                    pred_target[:, : key.size(1)],
                    normalize_anchor=self.additive_delta_normalization in {"states", "states_and_diff"},
                    normalize_output=self.additive_delta_normalization in {"states", "states_and_diff"},
                ).float()
            elif self.target_source == "contextual_delta" and pred_parallel is not None:
                query = _reconstruct_contextual_delta_next_state(
                    delta_anchor,
                    pred_target[:, : key.size(1)],
                    pred_parallel[:, : key.size(1)],
                ).float()
            elif self.target_source == "contextual_delta" and pred_magnitude is not None:
                scaled_delta = _scaled_contextual_delta(
                    pred_target[:, : key.size(1)],
                    delta_anchor,
                    pred_magnitude[:, : key.size(1)],
                )
                query = delta_anchor + scaled_delta
            else:
                query = delta_anchor + pred_target[:, : key.size(1)].float()
        else:
            if self.target_source == "rotation_scale_delta":
                _, contrastive_targets = self._rotation_scale_delta_targets(teacher_states, steps=self.contrastive_future_steps)
            elif self.target_source == "sphere_delta":
                _, contrastive_targets = self._sphere_delta_targets(teacher_states, steps=self.contrastive_future_steps)
            elif self.target_source == "additive_delta":
                _, contrastive_targets = self._additive_delta_targets(teacher_states, steps=self.contrastive_future_steps)
            else:
                _, contrastive_targets, _, _ = self._contextual_delta_targets(teacher_states, steps=self.contrastive_future_steps)
            query = pred_target[:, : contrastive_targets.size(1)].float()
            key = contrastive_targets.float()
        mask = payload_mask[:, : key.size(1)].reshape(-1)
        if int(mask.to(dtype=torch.int32).sum().item()) <= 1:
            return pred_target.new_zeros(())
        flat_query = query[:, : key.size(1)].reshape(-1, query.size(-2) * query.size(-1))[mask]
        flat_key = key.reshape(-1, key.size(-2) * key.size(-1))[mask]
        if flat_query.size(0) <= 1:
            return pred_target.new_zeros(())
        sample_count = min(self.contrastive_samples, flat_query.size(0))
        if sample_count < flat_query.size(0):
            sample_idx = torch.randperm(flat_query.size(0), device=flat_query.device)[:sample_count]
            flat_query = flat_query[sample_idx]
            flat_key = flat_key[sample_idx]
        query_repr = F.normalize(flat_query, dim=-1, eps=1e-8)
        key_repr = F.normalize(flat_key, dim=-1, eps=1e-8)
        logits = (query_repr @ key_repr.T) / self.contrastive_temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    def _predict_target_latents(
        self, byte_ids: Tensor
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
    ]:
        context_ids, target_ids = self._split_inputs_and_targets(byte_ids)
        context_latents = self.online_encoder(context_ids)
        if self.predictor_encoder is None:
            predictor_inputs = context_latents
        else:
            predictor_token_inputs = self.predictor_encoder(context_ids)
            if self.predictor_input == "tokens":
                predictor_inputs = predictor_token_inputs
            elif self.predictor_input == "context_plus_tokens":
                if self.predictor_fusion is None:
                    raise RuntimeError("Context-plus-tokens predictor requested without fusion module")
                fused_inputs = torch.cat([context_latents, predictor_token_inputs], dim=-1)
                predictor_inputs = context_latents + self.predictor_fusion(fused_inputs)
            else:
                raise RuntimeError(f"Unexpected predictor input mode {self.predictor_input!r}")
        action_latents: Tensor | None = None
        pred_parallel: Tensor | None = None
        pred_magnitude: Tensor | None = None
        if self.predictor_style == "action_transition":
            if self.action_head is None or self.transition_head is None:
                raise RuntimeError("Action-transition predictor requested without action/transition heads")
            action_latents = self.action_head(context_latents)
            state_features = predictor_inputs.unsqueeze(2).expand(-1, -1, self.num_target_codebooks, -1).float()
            action_features = action_latents.unsqueeze(2).expand(-1, -1, self.num_target_codebooks, -1).float()
            transition_inputs = torch.cat([state_features, action_features], dim=-1)
            pred_target = self.transition_head(transition_inputs)
        else:
            if self.predictor is None:
                raise RuntimeError("Direct predictor requested without predictor module")
            pred_target = self.predictor(predictor_inputs)
            pred_target = pred_target.view(*predictor_inputs.shape[:-1], self.num_target_codebooks, -1)
            if self.contextual_delta_parallel_head is not None:
                pred_parallel = self.contextual_delta_parallel_head(predictor_inputs).unsqueeze(-1)
            if self.contextual_delta_magnitude_head is not None:
                pred_magnitude = self.contextual_delta_magnitude_head(predictor_inputs).unsqueeze(-1)
        contextual_targets: Tensor | None = None
        teacher_states: Tensor | None = None
        target_parallel: Tensor | None = None
        target_magnitude: Tensor | None = None
        if self.target_source in {"contextual_only", "contextual_delta", "rotation_scale_delta", "sphere_delta", "additive_delta"}:
            contextual_states = self._encode_target_states(byte_ids)
            teacher_states = contextual_states
            if self.target_source == "contextual_only":
                contextual_targets = contextual_states[:, self.target_future_steps :, :].unsqueeze(2)
                target_latents = contextual_targets
            elif self.target_source == "contextual_delta":
                contextual_targets, target_latents, target_parallel, target_magnitude = self._contextual_delta_targets(
                    contextual_states,
                    steps=self.target_future_steps,
                )
            elif self.target_source == "sphere_delta":
                contextual_targets, sphere_delta_targets = self._sphere_delta_targets(
                    contextual_states,
                    steps=self.target_future_steps,
                )
                target_latents = contextual_targets if self.rotation_delta_loss_target == "next_state" else sphere_delta_targets
            elif self.target_source == "additive_delta":
                contextual_targets, additive_delta_targets = self._additive_delta_targets(
                    contextual_states,
                    steps=self.target_future_steps,
                )
                target_latents = contextual_targets if self.rotation_delta_loss_target == "next_state" else additive_delta_targets
            else:
                contextual_targets, rotation_delta_targets = self._rotation_scale_delta_targets(
                    contextual_states,
                    steps=self.target_future_steps,
                )
                target_latents = contextual_targets if self.rotation_delta_loss_target == "next_state" else rotation_delta_targets
        else:
            anchor_targets = self._anchor_targets(target_ids)
            target_latents = anchor_targets
            if self.target_encoder is not None or self.target_encoder_mode == "shared":
                contextual_states = self._encode_target_states(byte_ids)
                contextual_targets = contextual_states[:, 1:, :].unsqueeze(2)
                if self.contextual_target:
                    if self.contextual_residual_scale > 0.0:
                        target_latents = self._contextual_residual_targets(anchor_targets, contextual_targets)
                    else:
                        target_latents = (
                            self.target_anchor_weight * anchor_targets
                            + (1.0 - self.target_anchor_weight) * contextual_targets
                        )
        return (
            target_ids,
            context_latents,
            pred_target,
            pred_parallel,
            pred_magnitude,
            action_latents,
            target_latents,
            contextual_targets,
            teacher_states,
            target_parallel,
            target_magnitude,
        )

    def _per_position_loss(self, pred_target: Tensor, target_latents: Tensor) -> Tensor:
        pred_float = pred_target.float()
        target_float = target_latents.float()
        if self.jepa_loss == "cosine":
            pred_norm = F.normalize(pred_float, dim=-1, eps=1e-8)
            target_norm = F.normalize(target_float, dim=-1, eps=1e-8)
            per_book = 2.0 - 2.0 * (pred_norm * target_norm).sum(dim=-1)
            return per_book.mean(dim=-1)
        if self.jepa_loss == "mse":
            loss = F.mse_loss(pred_float, target_float, reduction="none").mean(dim=-1)
            return loss.mean(dim=-1)
        raise ValueError(f"Unknown JEPA_LOSS={self.jepa_loss!r}")

    def _representation_regularizer(self, pred_target: Tensor, payload_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.var_reg_weight <= 0 and self.cov_reg_weight <= 0:
            zero = pred_target.new_zeros(())
            return zero, zero, zero
        flat = pred_target.float()[payload_mask].reshape(-1, pred_target.size(-1))
        if flat.size(0) <= 1:
            zero = pred_target.new_zeros(())
            return zero, zero, zero
        centered = flat - flat.mean(dim=0, keepdim=True)
        std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
        var_loss = F.relu(1.0 - std).mean()
        cov = centered.T @ centered / max(flat.size(0) - 1, 1)
        cov_loss = _off_diagonal_mean_square(cov)
        reg = self.var_reg_weight * var_loss + self.cov_reg_weight * cov_loss
        return reg, var_loss.detach(), cov_loss.detach()

    def _target_representation_regularizer(self, target_latents: Tensor, payload_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.target_var_reg_weight <= 0 and self.target_cov_reg_weight <= 0:
            zero = target_latents.new_zeros(())
            return zero, zero, zero
        flat = target_latents.float()[payload_mask].reshape(-1, target_latents.size(-1))
        if flat.size(0) <= 1:
            zero = target_latents.new_zeros(())
            return zero, zero, zero
        centered = flat - flat.mean(dim=0, keepdim=True)
        std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
        var_loss = F.relu(1.0 - std).mean()
        cov = centered.T @ centered / max(flat.size(0) - 1, 1)
        cov_loss = _off_diagonal_mean_square(cov)
        reg = self.target_var_reg_weight * var_loss + self.target_cov_reg_weight * cov_loss
        return reg, var_loss.detach(), cov_loss.detach()

    @torch.no_grad()
    def _update_eval_codebooks(self, target_ids: Tensor, target_latents: Tensor, payload_mask: Tensor) -> None:
        if not self.uses_anchor_codebook or not self.contextual_target or self.contextual_residual_scale > 0.0:
            return
        flat_targets = target_ids[payload_mask]
        if flat_targets.numel() == 0:
            return
        masked_latents = target_latents.float()[payload_mask]
        decay = self.prototype_ema_decay
        for codebook_idx in range(self.num_target_codebooks):
            sums = torch.zeros_like(self.eval_codebooks[codebook_idx])
            counts = torch.zeros((self.vocab_size,), device=flat_targets.device, dtype=torch.float32)
            sums.index_add_(0, flat_targets, masked_latents[:, codebook_idx, :])
            counts.index_add_(0, flat_targets, torch.ones_like(flat_targets, dtype=torch.float32))
            valid = counts > 0
            if not valid.any():
                continue
            means = sums[valid] / counts[valid].unsqueeze(1)
            self.eval_codebooks[codebook_idx, valid].mul_(decay).add_(means, alpha=1.0 - decay)

    def _lm_probe_features(
        self,
        pred_target: Tensor,
        pred_parallel: Tensor | None,
        pred_magnitude: Tensor | None,
        context_latents: Tensor,
        action_latents: Tensor | None,
    ) -> Tensor:
        scaled_pred: Tensor | None = None
        if pred_magnitude is not None:
            scaled_pred = _scaled_contextual_delta(pred_target, context_latents.unsqueeze(2), pred_magnitude)
        if self.lm_probe_input == "pred_target":
            features = pred_target.float()
            return features.reshape(-1, self.num_target_codebooks * features.size(-1))
        if self.lm_probe_input == "pred_residual":
            features = _orthogonal_residual(pred_target, context_latents.unsqueeze(2))
            return features.reshape(-1, self.num_target_codebooks * features.size(-1))
        if self.lm_probe_input == "context_plus_pred":
            context_features = context_latents.unsqueeze(2).expand(-1, -1, self.num_target_codebooks, -1).float()
            features = torch.cat([context_features, pred_target.float()], dim=-1)
            return features.reshape(-1, self.num_target_codebooks * features.size(-1))
        if self.lm_probe_input == "context_plus_scaled_pred":
            if scaled_pred is None:
                raise RuntimeError("LM_PROBE_INPUT='context_plus_scaled_pred' requires contextual delta magnitude predictions")
            context_features = context_latents.unsqueeze(2).expand(-1, -1, self.num_target_codebooks, -1).float()
            features = torch.cat([context_features, scaled_pred.float()], dim=-1)
            return features.reshape(-1, self.num_target_codebooks * features.size(-1))
        if self.lm_probe_input == "context_plus_pred_parallel":
            if pred_parallel is None:
                raise RuntimeError("LM_PROBE_INPUT='context_plus_pred_parallel' requires contextual delta parallel predictions")
            context_features = context_latents.unsqueeze(2).expand(-1, -1, self.num_target_codebooks, -1).float()
            features = torch.cat([context_features, pred_target.float(), pred_parallel.float()], dim=-1)
            return features.reshape(-1, self.num_target_codebooks * features.size(-1))
        if self.lm_probe_input == "context_plus_pred_magnitude":
            if scaled_pred is None or pred_magnitude is None:
                raise RuntimeError("LM_PROBE_INPUT='context_plus_pred_magnitude' requires contextual delta magnitude predictions")
            context_features = context_latents.unsqueeze(2).expand(-1, -1, self.num_target_codebooks, -1).float()
            features = torch.cat([context_features, scaled_pred.float(), pred_magnitude.float()], dim=-1)
            return features.reshape(-1, self.num_target_codebooks * features.size(-1))
        if action_latents is None:
            raise RuntimeError(f"LM_PROBE_INPUT={self.lm_probe_input!r} requires action latents")
        if self.lm_probe_input == "action":
            return action_latents.float().reshape(-1, action_latents.size(-1))
        features = torch.cat([context_latents.float(), action_latents.float()], dim=-1)
        return features.reshape(-1, features.size(-1))

    def _lm_probe_logits(
        self,
        pred_target: Tensor,
        pred_parallel: Tensor | None,
        pred_magnitude: Tensor | None,
        context_latents: Tensor,
        action_latents: Tensor | None,
        *,
        detach_input: bool,
    ) -> Tensor:
        if self.lm_head is None:
            raise RuntimeError("LM probe requested without lm_head")
        flat_pred = self._lm_probe_features(pred_target, pred_parallel, pred_magnitude, context_latents, action_latents)
        if detach_input:
            flat_pred = flat_pred.detach()
        return self.lm_head(flat_pred)

    def _lm_probe_nll_sum(
        self,
        pred_target: Tensor,
        pred_parallel: Tensor | None,
        pred_magnitude: Tensor | None,
        context_latents: Tensor,
        action_latents: Tensor | None,
        target_ids: Tensor,
        payload_mask: Tensor,
        *,
        detach_input: bool,
    ) -> Tensor:
        if self.lm_head is None:
            return pred_target.new_zeros(())
        logits = self._lm_probe_logits(
            pred_target,
            pred_parallel,
            pred_magnitude,
            context_latents,
            action_latents,
            detach_input=detach_input,
        )
        flat_targets = target_ids.reshape(-1)
        nll = F.cross_entropy(logits, flat_targets, reduction="none")
        return nll[payload_mask.reshape(-1)].sum()

    def _validation_logits(self, pred_target: Tensor) -> Tensor:
        if not self.uses_anchor_codebook:
            raise RuntimeError("Validation logits without an LM head require a codebook-backed target source")
        flat_pred = pred_target.float().reshape(-1, self.num_target_codebooks, pred_target.size(-1))
        codebooks = self.eval_codebooks.detach().float()
        logits = torch.zeros((flat_pred.size(0), self.vocab_size), device=flat_pred.device, dtype=torch.float32)

        if self.eval_bridge == "cosine":
            pred_repr = F.normalize(flat_pred, dim=-1, eps=1e-8)
            codebook_repr = F.normalize(codebooks, dim=-1, eps=1e-8)
            for codebook_idx in range(self.num_target_codebooks):
                logits += pred_repr[:, codebook_idx, :] @ codebook_repr[codebook_idx].T
            return self.eval_logit_scale * logits

        if self.eval_bridge == "dot":
            for codebook_idx in range(self.num_target_codebooks):
                logits += flat_pred[:, codebook_idx, :] @ codebooks[codebook_idx].T
            return self.eval_logit_scale * logits

        if self.eval_bridge == "neg_l2":
            for codebook_idx in range(self.num_target_codebooks):
                pred_book = flat_pred[:, codebook_idx, :]
                codebook = codebooks[codebook_idx]
                pred_sq = (pred_book * pred_book).sum(dim=-1, keepdim=True)
                code_sq = (codebook * codebook).sum(dim=-1).unsqueeze(0)
                logits += -(pred_sq + code_sq - 2.0 * (pred_book @ codebook.T))
            return self.eval_logit_scale * logits

        if self.eval_bridge in {"mse", "neg_mse"}:
            for codebook_idx in range(self.num_target_codebooks):
                pred_book = flat_pred[:, codebook_idx, :]
                codebook = codebooks[codebook_idx]
                pred_sq = (pred_book * pred_book).sum(dim=-1, keepdim=True)
                code_sq = (codebook * codebook).sum(dim=-1).unsqueeze(0)
                sq_dist = pred_sq + code_sq - 2.0 * (pred_book @ codebook.T)
                logits += -(sq_dist / pred_book.size(-1))
            logits /= self.num_target_codebooks
            return self.eval_logit_scale * logits

        raise ValueError(f"Unknown EVAL_BRIDGE={self.eval_bridge!r}")

    def _compute_latent_metrics(
        self, byte_ids: Tensor
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        (
            target_ids,
            context_latents,
            pred_target,
            pred_parallel,
            pred_magnitude,
            action_latents,
            target_latents,
            contextual_targets,
            teacher_states,
            target_parallel,
            target_magnitude,
        ) = self._predict_target_latents(byte_ids)
        effective_len = target_latents.size(1)
        target_ids = target_ids[:, :effective_len]
        context_latents = context_latents[:, :effective_len]
        pred_target = pred_target[:, :effective_len]
        if pred_parallel is not None:
            pred_parallel = pred_parallel[:, :effective_len]
        if pred_magnitude is not None:
            pred_magnitude = pred_magnitude[:, :effective_len]
        if action_latents is not None:
            action_latents = action_latents[:, :effective_len]
        if contextual_targets is not None:
            contextual_targets = contextual_targets[:, :effective_len]
        if target_parallel is not None:
            target_parallel = target_parallel[:, :effective_len]
        if target_magnitude is not None:
            target_magnitude = target_magnitude[:, :effective_len]
        payload_mask = self._payload_mask(target_ids)
        num_bytes = payload_mask.to(dtype=torch.float32).sum()
        if int(num_bytes.item()) <= 0:
            raise ValueError("Batch contains no payload-byte targets")
        loss_pred = pred_target
        if self.target_source == "rotation_scale_delta" and self.rotation_delta_loss_target == "next_state":
            loss_pred = _reconstruct_rotation_scale_next_state(context_latents.unsqueeze(2), pred_target).to(dtype=pred_target.dtype)
        if self.target_source == "sphere_delta" and self.rotation_delta_loss_target == "next_state":
            loss_pred = _reconstruct_sphere_delta_next_state(context_latents.unsqueeze(2), pred_target).to(dtype=pred_target.dtype)
        if self.target_source == "additive_delta" and self.rotation_delta_loss_target == "next_state":
            loss_pred = _reconstruct_additive_delta_next_state(
                context_latents.unsqueeze(2),
                pred_target,
                normalize_anchor=self.additive_delta_normalization in {"states", "states_and_diff"},
                normalize_output=self.additive_delta_normalization in {"states", "states_and_diff"},
            ).to(dtype=pred_target.dtype)
        if self.target_source == "contextual_delta" and self.contextual_delta_magnitude_weight > 0.0:
            loss_pred = _orthogonal_residual(pred_target, context_latents.unsqueeze(2)).to(dtype=pred_target.dtype)
        per_pos_loss = self._per_position_loss(loss_pred, target_latents)
        loss_sum = per_pos_loss[payload_mask].sum()
        parallel_loss_sum = pred_target.new_zeros(())
        if self.contextual_delta_parallel_weight > 0.0:
            if pred_parallel is None or target_parallel is None:
                raise RuntimeError("Contextual delta parallel loss requested without scalar predictions/targets")
            parallel_loss = F.mse_loss(pred_parallel.float(), target_parallel.float(), reduction="none").mean(dim=-1).mean(dim=-1)
            parallel_loss_sum = parallel_loss[payload_mask].sum()
        magnitude_loss_sum = pred_target.new_zeros(())
        if self.contextual_delta_magnitude_weight > 0.0:
            if pred_magnitude is None or target_magnitude is None:
                raise RuntimeError("Contextual delta magnitude loss requested without scalar predictions/targets")
            magnitude_loss = F.mse_loss(pred_magnitude.float(), target_magnitude.float(), reduction="none").mean(dim=-1).mean(dim=-1)
            magnitude_loss_sum = magnitude_loss[payload_mask].sum()
        teacher_orth_loss_sum = pred_target.new_zeros(())
        if self.contextual_delta_teacher_orth_weight > 0.0:
            if teacher_states is None:
                raise RuntimeError("Teacher-orth loss requested without teacher states")
            teacher_anchor = teacher_states[:, :effective_len, :].unsqueeze(2)
            teacher_anchor_norm = F.normalize(teacher_anchor.float(), dim=-1, eps=1e-8)
            parallel_component = (pred_target.float() * teacher_anchor_norm).sum(dim=-1)
            teacher_orth_loss = parallel_component.square().mean(dim=-1)
            teacher_orth_loss_sum = teacher_orth_loss[payload_mask].sum()
        aux_loss_sum = pred_target.new_zeros(())
        if self.contextual_aux_weight > 0.0:
            if self.contextual_predictor is None or contextual_targets is None:
                raise RuntimeError("Contextual auxiliary loss requires a target encoder and contextual predictor")
            aux_pred = self.contextual_predictor(context_latents).unsqueeze(2)
            aux_per_pos_loss = self._per_position_loss(aux_pred, contextual_targets)
            aux_loss_sum = aux_per_pos_loss[payload_mask].sum()
        return (
            target_ids,
            payload_mask,
            num_bytes,
            context_latents.float(),
            pred_target.float(),
            pred_parallel.float() if pred_parallel is not None else None,
            pred_magnitude.float() if pred_magnitude is not None else None,
            action_latents.float() if action_latents is not None else None,
            target_latents.float(),
            teacher_states.float() if teacher_states is not None else None,
            loss_sum,
            parallel_loss_sum.float(),
            magnitude_loss_sum.float(),
            teacher_orth_loss_sum.float(),
            aux_loss_sum.float(),
        )

    def training_step(self, byte_ids: Tensor) -> dict[str, Tensor]:
        target_ids, payload_mask, num_bytes, context_latents, pred_target, pred_parallel, pred_magnitude, action_latents, target_latents, teacher_states, loss_sum, parallel_loss_sum, magnitude_loss_sum, teacher_orth_loss_sum, aux_loss_sum = (
            self._compute_latent_metrics(byte_ids)
        )
        base_loss = loss_sum / num_bytes
        parallel_loss = parallel_loss_sum / num_bytes
        magnitude_loss = magnitude_loss_sum / num_bytes
        teacher_orth_loss = teacher_orth_loss_sum / num_bytes
        aux_loss = aux_loss_sum / num_bytes
        reg_loss, var_loss, cov_loss = self._representation_regularizer(pred_target, payload_mask)
        target_reg_loss, target_var_loss, target_cov_loss = self._target_representation_regularizer(
            target_latents, payload_mask
        )
        contrastive_loss = self._contrastive_infonce_loss(pred_target, pred_parallel, pred_magnitude, context_latents, teacher_states, payload_mask)
        lm_probe_loss = self._lm_probe_nll_sum(
            pred_target,
            pred_parallel,
            pred_magnitude,
            context_latents,
            action_latents,
            target_ids,
            payload_mask,
            detach_input=True,
        ) / num_bytes
        loss = (
            base_loss
            + self.contextual_delta_parallel_weight * parallel_loss
            + self.contextual_delta_magnitude_weight * magnitude_loss
            + self.contextual_delta_teacher_orth_weight * teacher_orth_loss
            + self.contextual_aux_weight * aux_loss
            + reg_loss
            + target_reg_loss
            + self.contrastive_weight * contrastive_loss
            + self.lm_probe_weight * lm_probe_loss
        )
        self._update_eval_codebooks(target_ids, target_latents, payload_mask)
        return {
            "loss": loss,
            "loss_sum": loss_sum.detach(),
            "num_bytes": num_bytes.detach(),
            "parallel_loss": parallel_loss.detach(),
            "magnitude_loss": magnitude_loss.detach(),
            "teacher_orth_loss": teacher_orth_loss.detach(),
            "contextual_aux_loss": aux_loss.detach(),
            "lm_probe_loss": lm_probe_loss.detach(),
            "contrastive_loss": contrastive_loss.detach(),
            "repr_reg_loss": reg_loss.detach(),
            "var_loss": var_loss,
            "cov_loss": cov_loss,
            "target_repr_reg_loss": target_reg_loss.detach(),
            "target_var_loss": target_var_loss,
            "target_cov_loss": target_cov_loss,
        }

    def validation_step(self, byte_ids: Tensor) -> dict[str, Tensor]:
        target_ids, payload_mask, num_bytes, context_latents, pred_target, pred_parallel, pred_magnitude, action_latents, _, _, loss_sum, _, _, _, _ = self._compute_latent_metrics(byte_ids)
        if self.lm_head is not None:
            nll_sum_nat = self._lm_probe_nll_sum(
                pred_target,
                pred_parallel,
                pred_magnitude,
                context_latents,
                action_latents,
                target_ids,
                payload_mask,
                detach_input=False,
            )
        else:
            logits = self._validation_logits(pred_target)
            flat_targets = target_ids.reshape(-1)
            nll = F.cross_entropy(logits, flat_targets, reduction="none")
            nll_sum_nat = nll[payload_mask.reshape(-1)].sum()
        return {
            "loss_sum": loss_sum.detach(),
            "num_bytes": num_bytes.detach(),
            "nll_sum_nat": nll_sum_nat.detach(),
        }

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        decay = self.target_ema_decay
        if self.target_encoder is not None and self.target_encoder_mode == "ema":
            for tgt, src in zip(self.target_encoder.parameters(), self.online_encoder.parameters(), strict=True):
                tgt.data.mul_(decay).add_(src.data, alpha=1.0 - decay)
        if self.uses_anchor_codebook and self.target_codebook == "ema":
            self.target_codebooks[0].data.mul_(decay).add_(self.online_encoder.tok_emb.weight.data, alpha=1.0 - decay)
            if not self.contextual_target or self.contextual_residual_scale > 0.0:
                self.eval_codebooks[0].data.copy_(self.target_codebooks[0].data)

    def export_state_dict(self) -> dict[str, Tensor]:
        return self.state_dict()


def configure_optimizers(model: PureByteJEPA, args: Any) -> list[torch.optim.Optimizer]:
    optimizer_kind = getattr(args, "optimizer", "adamw").lower()
    named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]

    split_head = getattr(args, "split_lm_probe_optimizer", False) and model.lm_head is not None
    head_named_params = [(name, param) for name, param in named_params if split_head and name.startswith("lm_head.")]
    head_param_ids = {id(param) for _, param in head_named_params}
    main_named_params = [(name, param) for name, param in named_params if id(param) not in head_param_ids]

    if optimizer_kind == "muon":
        embed_params = [param for name, param in main_named_params if name.endswith("tok_emb.weight")]
        embed_param_ids = {id(param) for param in embed_params}
        matrix_params = [
            param for name, param in main_named_params if param.ndim == 2 and id(param) not in embed_param_ids
        ]
        matrix_param_ids = {id(param) for param in matrix_params}
        scalar_params = [
            param
            for _, param in main_named_params
            if id(param) not in embed_param_ids and id(param) not in matrix_param_ids
        ]

        optimizers: list[torch.optim.Optimizer] = []
        if embed_params:
            optimizer_embed = torch.optim.AdamW(
                [{"params": embed_params, "lr": args.embed_lr, "base_lr": args.embed_lr}],
                betas=(args.beta1, args.beta2),
                eps=args.adam_eps,
                weight_decay=args.weight_decay,
            )
            optimizers.append(optimizer_embed)
        if matrix_params:
            optimizer_muon = Muon(
                matrix_params,
                lr=args.matrix_lr,
                momentum=args.muon_momentum,
                backend_steps=args.muon_backend_steps,
            )
            for group in optimizer_muon.param_groups:
                group["base_lr"] = args.matrix_lr
            optimizers.append(optimizer_muon)
        if scalar_params:
            optimizer_scalar = torch.optim.AdamW(
                [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
                betas=(args.beta1, args.beta2),
                eps=args.adam_eps,
                weight_decay=args.weight_decay,
            )
            optimizers.append(optimizer_scalar)
        if head_named_params:
            optimizer_head = torch.optim.AdamW(
                [{"params": [param for _, param in head_named_params], "lr": args.lm_probe_lr, "base_lr": args.lm_probe_lr}],
                betas=(args.beta1, args.beta2),
                eps=args.adam_eps,
                weight_decay=args.weight_decay,
            )
            optimizers.append(optimizer_head)
        return optimizers

    if split_head:
        lm_head_params = [param for _, param in head_named_params]
        main_params = [param for _, param in main_named_params]
        main_optimizer = torch.optim.AdamW(
            main_params,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
        probe_optimizer = torch.optim.AdamW(
            lm_head_params,
            lr=args.lm_probe_lr,
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
        return [main_optimizer, probe_optimizer]

    optimizer = torch.optim.AdamW(
        [param for _, param in named_params],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    return [optimizer]
