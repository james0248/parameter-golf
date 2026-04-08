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
        target_anchor_weight: float,
        prototype_ema_decay: float,
        contextual_residual_scale: float,
        contextual_aux_weight: float,
        lm_probe_weight: float,
        lm_probe_input: str,
        var_reg_weight: float,
        cov_reg_weight: float,
    ):
        super().__init__()
        if num_target_codebooks <= 0:
            raise ValueError(f"NUM_TARGET_CODEBOOKS must be positive, got {num_target_codebooks}")
        if predictor_input not in {"context_latent", "tokens"}:
            raise ValueError(f"Unknown PREDICTOR_INPUT={predictor_input!r}")
        if target_source not in {"codebook", "contextual_only", "contextual_delta"}:
            raise ValueError(f"Unknown TARGET_SOURCE={target_source!r}")
        if lm_probe_input not in {"pred_target", "pred_residual"}:
            raise ValueError(f"Unknown LM_PROBE_INPUT={lm_probe_input!r}")
        if contextual_target and num_target_codebooks != 1:
            raise ValueError("CONTEXTUAL_TARGET=1 currently requires NUM_TARGET_CODEBOOKS=1")
        if target_source in {"contextual_only", "contextual_delta"} and num_target_codebooks != 1:
            raise ValueError("TARGET_SOURCE contextual variants currently require NUM_TARGET_CODEBOOKS=1")
        if target_codebook == "ema" and num_target_codebooks != 1:
            raise ValueError("TARGET_CODEBOOK='ema' only supports NUM_TARGET_CODEBOOKS=1")
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
        self.target_anchor_weight = float(target_anchor_weight)
        self.prototype_ema_decay = float(prototype_ema_decay)
        self.contextual_residual_scale = float(contextual_residual_scale)
        self.contextual_aux_weight = float(contextual_aux_weight)
        self.lm_probe_weight = float(lm_probe_weight)
        self.lm_probe_input = lm_probe_input
        self.var_reg_weight = float(var_reg_weight)
        self.cov_reg_weight = float(cov_reg_weight)
        self.predictor_input = predictor_input
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
        if self.contextual_target or self.contextual_aux_weight > 0.0 or self.target_source in {"contextual_only", "contextual_delta"}:
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
        if self.predictor_input == "tokens":
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
        self.predictor = Predictor(
            model_dim=model_dim,
            hidden_dim=predictor_hidden_dim,
            output_dim=model_dim * self.num_target_codebooks,
            num_layers=predictor_num_layers,
            num_heads=predictor_num_heads,
            dropout=dropout,
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
            self.lm_head = nn.Sequential(
                nn.LayerNorm(model_dim * self.num_target_codebooks),
                nn.Linear(model_dim * self.num_target_codebooks, vocab_size),
            )

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

    def _contextual_delta_targets(self, contextual_states: Tensor) -> tuple[Tensor, Tensor]:
        prev_targets = contextual_states[:, :-1, :].unsqueeze(2)
        next_targets = contextual_states[:, 1:, :].unsqueeze(2)
        delta_targets = _orthogonal_residual(next_targets, prev_targets)
        delta_targets = F.normalize(delta_targets, dim=-1, eps=1e-8)
        return next_targets, delta_targets.to(dtype=next_targets.dtype)

    def _predict_target_latents(self, byte_ids: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
        context_ids, target_ids = self._split_inputs_and_targets(byte_ids)
        context_latents = self.online_encoder(context_ids)
        predictor_inputs = context_latents if self.predictor_encoder is None else self.predictor_encoder(context_ids)
        pred_target = self.predictor(predictor_inputs)
        pred_target = pred_target.view(*predictor_inputs.shape[:-1], self.num_target_codebooks, -1)
        contextual_targets: Tensor | None = None
        with torch.no_grad():
            if self.target_source in {"contextual_only", "contextual_delta"}:
                if self.target_encoder is None:
                    raise RuntimeError("Contextual target sources require a target encoder")
                contextual_states = self.target_encoder(byte_ids)
                if self.target_source == "contextual_only":
                    contextual_targets = contextual_states[:, 1:, :].unsqueeze(2)
                    target_latents = contextual_targets
                else:
                    contextual_targets, target_latents = self._contextual_delta_targets(contextual_states)
            else:
                anchor_targets = self._anchor_targets(target_ids)
                target_latents = anchor_targets
                if self.target_encoder is not None:
                    contextual_targets = self.target_encoder(byte_ids)[:, 1:, :].unsqueeze(2)
                    if self.contextual_target:
                        if self.contextual_residual_scale > 0.0:
                            target_latents = self._contextual_residual_targets(anchor_targets, contextual_targets)
                        else:
                            target_latents = (
                                self.target_anchor_weight * anchor_targets
                                + (1.0 - self.target_anchor_weight) * contextual_targets
                            )
        return target_ids, context_latents, pred_target, target_latents, contextual_targets

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

    def _lm_probe_features(self, pred_target: Tensor, context_latents: Tensor) -> Tensor:
        if self.lm_probe_input == "pred_target":
            features = pred_target.float()
        else:
            features = _orthogonal_residual(pred_target, context_latents.unsqueeze(2))
        return features.reshape(-1, self.num_target_codebooks * features.size(-1))

    def _lm_probe_logits(self, pred_target: Tensor, context_latents: Tensor, *, detach_input: bool) -> Tensor:
        if self.lm_head is None:
            raise RuntimeError("LM probe requested without lm_head")
        flat_pred = self._lm_probe_features(pred_target, context_latents)
        if detach_input:
            flat_pred = flat_pred.detach()
        return self.lm_head(flat_pred)

    def _lm_probe_nll_sum(
        self,
        pred_target: Tensor,
        context_latents: Tensor,
        target_ids: Tensor,
        payload_mask: Tensor,
        *,
        detach_input: bool,
    ) -> Tensor:
        if self.lm_head is None:
            return pred_target.new_zeros(())
        logits = self._lm_probe_logits(pred_target, context_latents, detach_input=detach_input)
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
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        target_ids, context_latents, pred_target, target_latents, contextual_targets = self._predict_target_latents(byte_ids)
        payload_mask = self._payload_mask(target_ids)
        num_bytes = payload_mask.to(dtype=torch.float32).sum()
        if int(num_bytes.item()) <= 0:
            raise ValueError("Batch contains no payload-byte targets")
        per_pos_loss = self._per_position_loss(pred_target, target_latents)
        loss_sum = per_pos_loss[payload_mask].sum()
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
            target_latents.float(),
            loss_sum,
            aux_loss_sum.float(),
        )

    def training_step(self, byte_ids: Tensor) -> dict[str, Tensor]:
        target_ids, payload_mask, num_bytes, context_latents, pred_target, target_latents, loss_sum, aux_loss_sum = (
            self._compute_latent_metrics(byte_ids)
        )
        base_loss = loss_sum / num_bytes
        aux_loss = aux_loss_sum / num_bytes
        reg_loss, var_loss, cov_loss = self._representation_regularizer(pred_target, payload_mask)
        lm_probe_loss = self._lm_probe_nll_sum(
            pred_target,
            context_latents,
            target_ids,
            payload_mask,
            detach_input=True,
        ) / num_bytes
        loss = base_loss + self.contextual_aux_weight * aux_loss + reg_loss + self.lm_probe_weight * lm_probe_loss
        self._update_eval_codebooks(target_ids, target_latents, payload_mask)
        return {
            "loss": loss,
            "loss_sum": loss_sum.detach(),
            "num_bytes": num_bytes.detach(),
            "contextual_aux_loss": aux_loss.detach(),
            "lm_probe_loss": lm_probe_loss.detach(),
            "repr_reg_loss": reg_loss.detach(),
            "var_loss": var_loss,
            "cov_loss": cov_loss,
        }

    def validation_step(self, byte_ids: Tensor) -> dict[str, Tensor]:
        target_ids, payload_mask, num_bytes, context_latents, pred_target, _, loss_sum, _ = self._compute_latent_metrics(byte_ids)
        if self.lm_head is not None:
            nll_sum_nat = self._lm_probe_nll_sum(
                pred_target,
                context_latents,
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
        if self.target_encoder is not None:
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
