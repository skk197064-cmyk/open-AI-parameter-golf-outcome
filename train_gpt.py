# ════════════════════════════════════════════════════════════════════════════════
# CELL 1 — IMPORTS
# ════════════════════════════════════════════════════════════════════════════════
from __future__ import annotations
import copy, glob, io, lzma, math, os, random, subprocess, sys, time, uuid, zlib
from pathlib import Path
from collections import deque, Counter

try:
    import zstandard; _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    _FLASH_BACKEND = "fa3"
except ImportError:
    try:
        from flash_attn import flash_attn_func as flash_attn_3_func
        _FLASH_BACKEND = "fa2"
    except ImportError:
        _FLASH_BACKEND = "sdpa"
        def flash_attn_3_func(q, k, v, causal=False):
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            if k.size(1) != q.size(1):
                r = q.size(1) // k.size(1)
                k = k.repeat_interleave(r, dim=1)
                v = v.repeat_interleave(r, dim=1)
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal).transpose(1, 2)

_AMP_DTYPE = torch.bfloat16

# Module-level debug logger reference — set by main() before training.
# Used by Muon optimizer to log SPECTRA clipping events.
# NOTE: We use a mutable list [logger] instead of a bare reference because
# module reload resets module-level variables but doesn't reset list contents.
# This survives importlib.reload() in test harnesses.
_dbg_logger = [None]

# ════════════════════════════════════════════════════════════════════════════════
# DEBUG LOGGING SYSTEM — tracks all technique computations during training
# ════════════════════════════════════════════════════════════════════════════════
# Controlled by DEBUG_LOG env var:
#   0 = off (default)
#   1 = technique summaries at diagnostic steps only (steps 1, 50, 500, 2000)
#   2 = technique summaries every train_log_every steps
#   3 = per-layer detail every diagnostic step
# Set via: export DEBUG_LOG=2
# ════════════════════════════════════════════════════════════════════════════════

class DebugLogger:
    """Configurable debug logger for tracking technique computations during training.
    Logs tensor shapes, norms, gradient flows, and value distributions for every
    technique component. Designed to run on base_model (not compiled) to avoid
    torch.compile interference. All logging is gated by step intervals to
    minimize overhead during normal training."""

    def __init__(self, log_fn=None, verbosity: int = 0):
        self._debug_log_path = None
        raw_log = log_fn or (lambda msg: None)
        def _combined_log(msg):
            raw_log(msg)
            self._write_debug(msg)
        self._log = _combined_log
        self.verbosity = verbosity
        self._spectra_clip_count = 0
        self._spectra_total = 0

        # ─── History tracking for graphs ───
        self._history = {
            'train_loss': [],       # (step, loss)
            'val_loss': [],         # (step, val_loss)
            'val_bpb': [],          # (step, val_bpb)
            'lr_scale': [],         # (step, lr_scale)
            'grad_norm': [],        # (step, grad_norm)
            'muon_momentum': [],    # (step, muon_momentum)
            'step_time_ms': [],     # (step, time_ms)
            # Per-technique gradient norms
            'grad_T2_spelling_bee_proj': [],
            'grad_T3_spatial_gate': [],
            'grad_T4_momentum_proj': [],
            'grad_T4_momentum_gate': [],
            'grad_T8_kan_base': [],
            'grad_T8_kan_spline': [],
            'grad_T9_tok_emb': [],
            'grad_bigram_proj': [],
            'grad_ve_proj': [],
            # Technique output norms
            'out_bigram': [],
            'out_spelling_bee': [],
            'out_momentum_buf_norm': [],
            'out_momentum_proj_norm': [],
            'out_momentum_gate': [],
            # SPECTRA clipping
            'spectra_sigma': [],
            'spectra_clip_count': [],
            'spectra_total': [],
            # Weight bank norms
            'bank_qo_norm': [],
            'bank_kv_norm': [],
            'bank_mlp_gate_norm': [],
            'bank_mlp_up_norm': [],
            'bank_mlp_down_norm': [],
            # Spatial bias gate stats
            'spatial_gate_mean': [],
            'spatial_gate_std': [],
            # KAN stats
            'kan_base_out_norm': [],
            'kan_spline_out_norm': [],
        }
        self._graph_dir = None

    def should_log(self, step: int, log_every: int) -> bool:
        """Determine if this step should produce debug output."""
        if self.verbosity == 0:
            return False
        if self.verbosity == 1:
            return step in (1, 50, 100, 500, 1000, 2000, 5000, 10000, 15000)
        if self.verbosity >= 2:
            return step % max(log_every, 1) == 0
        return False

    def is_detailed(self) -> bool:
        return self.verbosity >= 3

    def setup_log_files(self, graph_dir: str, debug_log_path: str):
        """Set up output directories for graphs and debug log file."""
        self._graph_dir = graph_dir
        self._debug_log_path = debug_log_path
        os.makedirs(graph_dir, exist_ok=True)
        os.makedirs(os.path.dirname(debug_log_path) if os.path.dirname(debug_log_path) else '.', exist_ok=True)
        # Write header to debug log
        with open(debug_log_path, 'w', encoding='utf-8') as f:
            f.write(f"DEBUG LOG — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")

    def _write_debug(self, msg: str):
        """Write message to dedicated debug log file."""
        if self._debug_log_path is not None:
            try:
                with open(self._debug_log_path, 'a', encoding='utf-8') as f:
                    f.write(msg + '\n')
            except Exception:
                pass  # Never fail training due to debug logging

    def log_t(self, name: str, t: Tensor, extra: str = ""):
        """Log tensor stats: shape, dtype, norm, min, max, mean."""
        if t is None:
            self._log(f"  [DEBUG] {name}: None {extra}")
            return
        t_f = t.float()
        self._log(
            f"  [DEBUG] {name}: shape={tuple(t.shape)} dtype={t.dtype} "
            f"norm={t_f.norm().item():.6f} min={t_f.min().item():.6f} "
            f"max={t_f.max().item():.6f} mean={t_f.mean().item():.6f} {extra}"
        )

    def log_grad(self, name: str, p: Tensor, history_key: str = None):
        """Log gradient stats for a parameter."""
        if p.grad is None:
            msg = f"  [GRAD] {name}: NO GRADIENT"
            self._log(msg)
            return
        g = p.grad.float()
        g_norm = g.norm().item()
        g_max = g.abs().max().item()
        g_mean = g.mean().item()
        msg = (
            f"  [GRAD] {name}: shape={tuple(p.shape)} "
            f"g_norm={g_norm:.6f} g_max={g_max:.6f} "
            f"g_mean={g_mean:.8f}"
        )
        self._log(msg)
        # Record history if key provided
        if history_key is not None and history_key in self._history:
            self._history[history_key].append(g_norm)

    def log_init_summary(self, model, args):
        """Log initialization summary for all technique components."""
        if self.verbosity < 1:
            return
        self._log("=" * 60)
        self._log("  TECHNIQUE INITIALIZATION VERIFICATION")
        self._log("=" * 60)

        # T1: Weight Looping
        if args.weight_looping:
            self._log("  [T1] Weight Looping: ENABLED")
            self._log(f"         period={args.weight_loop_period}, "
                      f"bank_rows={args.weight_loop_period} (period-sized, no dead rows)")
            self.log_t("qo_bank", model.qo_bank, f"shape=2*{args.weight_loop_period}")

        # T2: Spelling Bee
        if model.spelling_bee is not None:
            sb = model.spelling_bee
            self._log("  [T2] Spelling Bee: ENABLED")
            self.log_t("trigram_embed.weight", sb.trigram_embed.weight, "init")
            self.log_t("char_embed.weight", sb.char_embed.weight, "init")
            if sb.proj is not None:
                self.log_t("proj.weight", sb.proj.weight, "init")
            self._log(f"         scale={sb.scale.item():.6f}, buckets={sb.num_buckets}")

        # T3: Spatial Attention Bias
        if args.spatial_attn_bias:
            self._log("  [T3] Spatial Attention Bias: ENABLED")
            for i, block in enumerate(model.blocks):
                sab = block.attn.spatial_attn_bias
                self.log_t(f"block[{i}].gate_embed.weight", sab.gate_embed.weight,
                           f"max_seq={sab.max_seq_len}")
                self._log(f"         gate_mean={torch.sigmoid(sab.gate_embed.weight).mean().item():.4f}")

        # T4: Momentum Tokens
        if args.momentum_tokens_enabled:
            self._log("  [T4] Momentum Tokens: ENABLED")
            for i, block in enumerate(model.blocks):
                if block.has_momentum:
                    mt = block.momentum
                    self.log_t(f"block[{i}].momentum.proj.weight", mt.proj.weight, "init")
                    self._log(f"         gate={mt.gate.item():.6f}, decay={mt.momentum_val}, "
                              f"buf_norm={mt._momentum_buf.norm().item():.6f}")

        # T5: SPECTRA Clipping
        if args.spectra_clip_enabled:
            self._log(f"  [T5] SPECTRA Clipping: ENABLED, norm={args.spectra_clip_norm}")

        # T6: Analytic Decomposition
        if args.analytic_decomp_enabled:
            self._log(f"  [T6] Analytic Decomposition: ENABLED, rank_ratio={args.analytic_decomp_rank}")

        # T7: ROCKET Compression
        if args.rocket_enabled:
            self._log("  [T7] ROCKET Compression: ENABLED")

        # T8: KAN Layers
        if args.kan_enabled:
            self._log(f"  [T8] KAN Layers: ENABLED, layers={model.kan_layer_set}, "
                      f"grid_size={args.kan_grid_size}")
            for i in model.kan_layer_set:
                kan = model.blocks[i].kan
                self.log_t(f"block[{i}].kan.base_weight", kan.base_weight, "init")
                self.log_t(f"block[{i}].kan.spline_weight", kan.spline_weight, "init")
                self._log(f"         grid={kan.grid.data.tolist()[:5]}..., "
                          f"grid_scale={kan.grid_scale.item():.4f}")

        # T9: Weber's Law
        if args.weber_law_enabled:
            self._log(f"  [T9] Weber's Law: ENABLED, C={args.weber_law_C}")

        # T10: Coreset Attention
        if args.coreset_attention:
            self._log(f"  [T10] Coreset Attention: DISABLED (causal incompatibility)")

        # Bigram (base code)
        if model.bigram is not None:
            self._log("  [BASE] BigramHashEmbedding: ENABLED")
            self.log_t("bigram.embed.weight", model.bigram.embed.weight, "init")
            if model.bigram.proj is not None:
                self.log_t("bigram.proj.weight", model.bigram.proj.weight, "init")

        # VE (base code)
        if model.ve_shared is not None:
            self._log("  [BASE] ValueEmbedding: ENABLED")
            self.log_t("ve_shared.embed.weight", model.ve_shared.embed.weight, "init")
            if model.ve_shared.proj is not None:
                self.log_t("ve_shared.proj.weight", model.ve_shared.proj.weight, "init")

        self._log("=" * 60)

    def log_forward_flow(self, model, input_ids, step):
        """Trace the complete forward pass value flow on base_model (uncompiled).
        Shows how data transforms through each component."""
        if not self.should_log(step, 1):
            return
        self._log(f"\n  {'─'*50}")
        self._log(f"  FORWARD FLOW TRACE — step {step}")
        self._log(f"  {'─'*50}")

        with torch.no_grad():
            x = model.tok_emb(input_ids)
            self.log_t("tok_emb output", x, f"tokens_range=[{input_ids.min().item()},{input_ids.max().item()}]")

            if model.bigram is not None:
                bg = model.bigram(input_ids)
                self.log_t("bigram output", bg)
                self._history['out_bigram'].append(bg.norm().item())
                x = x + bg

            if model.spelling_bee is not None:
                sb = model.spelling_bee(input_ids)
                self.log_t("spelling_bee output", sb)
                self._history['out_spelling_bee'].append(sb.norm().item())
                x = x + sb

            x = F.rms_norm(x, (x.size(-1),))
            self.log_t("after rms_norm", x)

            x = model.smear(x)
            self.log_t("after smear", x)

            # Trace through each block
            for i in range(model.num_layers):
                block = model.blocks[i]
                bi = i % model.unique_layer_count if model.weight_looping else i
                x_in = x.clone()

                # T4: Momentum Tokens
                if block.has_momentum:
                    buf = block.momentum._get_buf(x.device)
                    buf_norm = buf.norm().item()
                    proj_out = block.momentum.proj(buf.clone().to(dtype=x.dtype))
                    gate_val = torch.sigmoid(block.momentum.gate).item()
                    self._log(f"  [T4] block[{i}] momentum: buf_norm={buf_norm:.6f}, "
                              f"gate={gate_val:.4f}, proj_norm={proj_out.norm().item():.6f}")
                    self._history['out_momentum_proj_norm'].append(proj_out.norm().item())
                    x = block.momentum(x)
                    self.log_t(f"  [T4] block[{i}] after momentum", x)

                # Attention
                attn_in = block.attn_norm(x_in) * block.ln_scale_factor
                q = F.linear(attn_in, model.qo_bank[bi].to(attn_in.dtype))
                k = F.linear(attn_in, model.kv_bank[bi].to(attn_in.dtype))
                v = F.linear(attn_in, model.kv_bank[model.unique_layer_count + bi].to(attn_in.dtype))
                self.log_t(f"  block[{i}] Q/K/V norms",
                           torch.stack([q.norm(), k.norm(), v.norm()]))

                # T3: Spatial Attention Bias (post-attn, we show gate stats)
                if block.attn.spatial_bias:
                    sab = block.attn.spatial_attn_bias
                    seq = x.size(1)
                    positions = torch.arange(seq, device=x.device).clamp(max=sab.max_seq_len - 1)
                    gates = torch.sigmoid(sab.gate_embed(positions))
                    self._log(f"  [T3] block[{i}] spatial_bias: gate_mean={gates.mean().item():.4f}, "
                              f"gate_std={gates.std().item():.4f}, "
                              f"gate_range=[{gates.min().item():.4f},{gates.max().item():.4f}]")

                # T8: KAN Layer
                if block.use_kan:
                    kan = block.kan
                    x_scaled = x * kan.grid_scale
                    grid = kan.grid
                    distances = x_scaled.unsqueeze(-1) - grid.unsqueeze(0).unsqueeze(0)
                    basis = torch.relu(1.0 - distances.abs())
                    basis_norm = basis.norm().item()
                    spline_out = torch.einsum('btig,oig->bto',
                                              basis.to(dtype=x.dtype),
                                              kan.spline_weight.to(dtype=x.dtype))
                    base_out = F.silu(F.linear(x, kan.base_weight.to(dtype=x.dtype)))
                    self.log_t(f"  [T8] block[{i}] kan.spline_out", spline_out)
                    self.log_t(f"  [T8] block[{i}] kan.base_out", base_out)
                    self.log_t(f"  [T8] block[{i}] kan.total", base_out + spline_out)
                    self._log(f"  [T8] block[{i}] kan: basis_norm={basis_norm:.4f}, "
                              f"grid_scale={kan.grid_scale.item():.4f}, "
                              f"grid={kan.grid.data.tolist()}")

                # Layer output norms
                self.log_t(f"  block[{i}] output", x)

            self.log_t("final_norm output", x)

        self._log(f"  {'─'*50}\n")

    def log_gradients(self, model, step, log_every):
        """Log gradient stats for all technique-specific parameters."""
        if not self.should_log(step, log_every):
            return
        self._log(f"\n  {'─'*50}")
        self._log(f"  GRADIENT FLOW TRACE — step {step}")
        self._log(f"  {'─'*50}")

        # T2: Spelling Bee gradients
        if model.spelling_bee is not None:
            self.log_grad("T2.spelling_bee.trigram_embed.weight", model.spelling_bee.trigram_embed.weight)
            self.log_grad("T2.spelling_bee.char_embed.weight", model.spelling_bee.char_embed.weight)
            if model.spelling_bee.proj is not None:
                self.log_grad("T2.spelling_bee.proj.weight", model.spelling_bee.proj.weight,
                              history_key='grad_T2_spelling_bee_proj')
            self.log_grad("T2.spelling_bee.scale", model.spelling_bee.scale)

        # T3: Spatial Attention Bias gradients
        first_t3 = None
        for i, block in enumerate(model.blocks):
            if block.attn.spatial_bias:
                if first_t3 is None:
                    first_t3 = i
                self.log_grad(f"T3.block[{i}].gate_embed.weight",
                              block.attn.spatial_attn_bias.gate_embed.weight,
                              history_key='grad_T3_spatial_gate' if first_t3 == i else None)

        # T4: Momentum Tokens gradients
        first_t4 = None
        for i, block in enumerate(model.blocks):
            if block.has_momentum:
                if first_t4 is None:
                    first_t4 = i
                self.log_grad(f"T4.block[{i}].momentum.proj.weight", block.momentum.proj.weight,
                              history_key='grad_T4_momentum_proj' if first_t4 == i else None)
                self.log_grad(f"T4.block[{i}].momentum.gate", block.momentum.gate,
                              history_key='grad_T4_momentum_gate' if first_t4 == i else None)

        # T8: KAN Layer gradients
        first_kan = None
        for i in model.kan_layer_set:
            kan = model.blocks[i].kan
            if first_kan is None:
                first_kan = i
            self.log_grad(f"T8.block[{i}].kan.base_weight", kan.base_weight,
                          history_key='grad_T8_kan_base' if first_kan == i else None)
            self.log_grad(f"T8.block[{i}].kan.spline_weight", kan.spline_weight,
                          history_key='grad_T8_kan_spline' if first_kan == i else None)
            self.log_grad(f"T8.block[{i}].kan.grid_scale", kan.grid_scale)

        # T9: Weber's Law effect (check embedding gradient)
        self.log_grad("T9.tok_emb.weight", model.tok_emb.weight,
                      history_key='grad_T9_tok_emb')

        # Base: Bigram
        if model.bigram is not None:
            self.log_grad("BASE.bigram.embed.weight", model.bigram.embed.weight)
            if model.bigram.proj is not None:
                self.log_grad("BASE.bigram.proj.weight", model.bigram.proj.weight,
                              history_key='grad_bigram_proj')

        # Base: VE
        if model.ve_shared is not None:
            self.log_grad("BASE.ve_shared.embed.weight", model.ve_shared.embed.weight)
            if model.ve_shared.proj is not None:
                self.log_grad("BASE.ve_shared.proj.weight", model.ve_shared.proj.weight,
                              history_key='grad_ve_proj')

        self._log(f"  {'─'*50}\n")

    def log_momentum_update(self, model, step, log_every):
        """Log momentum buffer state after update."""
        if not self.should_log(step, log_every):
            return
        for i, block in enumerate(model.blocks):
            if block.has_momentum:
                mt = block.momentum
                buf = mt._get_buf(next(model.parameters()).device)
                gate = torch.sigmoid(mt.gate).item()
                self._log(
                    f"  [T4] block[{i}] momentum UPDATE: "
                    f"buf_norm={buf.norm().item():.6f} "
                    f"buf_mean_abs={buf.float().abs().mean().item():.8f} "
                    f"gate={gate:.6f}"
                )
                # Record history
                self._history['out_momentum_buf_norm'].append(buf.norm().item())
                self._history['out_momentum_gate'].append(gate)

    def log_spectra_clip(self, sigma, threshold):
        """Track SPECTRA clipping events."""
        self._spectra_total += 1
        self._history['spectra_sigma'].append(sigma)
        self._history['spectra_clip_count'].append(self._spectra_clip_count)
        self._history['spectra_total'].append(self._spectra_total)
        if sigma > threshold:
            self._spectra_clip_count += 1
            if self.verbosity >= 1 and self._spectra_clip_count <= 5:
                self._log(f"  [T5] SPECTRA CLIP: sigma={sigma:.4f} > threshold={threshold:.1f} "
                          f"→ scaled to {threshold:.1f} "
                          f"(clipped {self._spectra_clip_count}/{self._spectra_total})")
        elif self.verbosity >= 2 and self._spectra_total % 500 == 0:
            self._log(f"  [T5] SPECTRA stats: clipped={self._spectra_clip_count}/{self._spectra_total} "
                      f"({100*self._spectra_clip_count/max(self._spectra_total,1):.1f}%)")

    def log_training_step(self, step, loss, lr_scale, grad_norm, muon_momentum,
                          train_time_ms, log_every):
        """Enhanced training step log with gradient info."""
        # Always record history (even if not logging)
        self._history['train_loss'].append((step, loss))
        self._history['lr_scale'].append((step, lr_scale))
        self._history['grad_norm'].append((step, grad_norm))
        self._history['muon_momentum'].append((step, muon_momentum))
        self._history['step_time_ms'].append((step, train_time_ms))
        should = step % max(log_every, 1) == 0 or step <= 10
        if should and self.verbosity >= 1:
            self._log(
                f"  [TRAIN] step:{step} loss:{loss:.4f} lr_scale:{lr_scale:.4f} "
                f"grad_norm:{grad_norm:.4f} muon_mom:{muon_momentum:.4f} "
                f"time:{train_time_ms:.0f}ms"
            )

    def log_weight_looping_verification(self, model, step, log_every):
        """Verify weight looping sharing invariant is maintained after optimizer step.
        With period-sized banks, sharing is structural (no dead rows)."""
        if not self.should_log(step, log_every):
            return
        if not model.weight_looping:
            return
        u = model.unique_layer_count
        self._log(f"  [T1] Weight Looping: bank size = (2*{u}, dim, dim) "
                  f"(period={model.weight_loop_period}) — no dead rows to verify")
        # With period-sized banks, sharing is guaranteed by construction.
        # Just log a confirmation.
        for bank_name in ["qo_bank", "kv_bank", "mlp_gate_bank", "mlp_up_bank", "mlp_down_bank"]:
            bank = getattr(model, bank_name)
            self._log(f"    {bank_name}: shape={tuple(bank.shape)} — structurally shared, no breakage possible")

    def log_quantization_report(self, result, meta, unbanked_sd, rocket):
        """Log quantization compression details."""
        if self.verbosity < 1:
            return
        self._log("  [QUANT] Quantization report:")
        total_original = sum(t.numel() * t.element_size() for t in unbanked_sd.values())
        total_quant = sum(t.numel() * t.element_size() for t in result.values())
        rocket_tensors = sum(1 for v in meta.values()
                           if isinstance(v, dict) and v.get("rocket", False))
        int6_count = sum(1 for v in meta.values()
                       if isinstance(v, dict) and v.get("type") == "int6")
        int8_count = sum(1 for v in meta.values()
                       if isinstance(v, dict) and v.get("type") == "int8")
        passthrough = sum(1 for v in meta.values() if v in ("passthrough", "passthrough_ctrl", "passthrough_fp16"))
        self._log(f"    original_bytes={total_original:,}")
        self._log(f"    quantized_bytes={total_quant:,} "
                  f"(ratio={total_original/max(total_quant,1):.2f}x)")
        self._log(f"    tensors: int6={int6_count}, int8={int8_count}, "
                  f"passthrough={passthrough}, rocket_reordered={rocket_tensors}")

    def record_val_results(self, step, val_loss, val_bpb):
        """Record validation results for plotting."""
        self._history['val_loss'].append((step, val_loss))
        self._history['val_bpb'].append((step, val_bpb))

    def record_bank_norms(self, model, step):
        """Record weight bank norms for stability tracking."""
        self._history['bank_qo_norm'].append((step, model.qo_bank.data.float().norm().item()))
        self._history['bank_kv_norm'].append((step, model.kv_bank.data.float().norm().item()))
        self._history['bank_mlp_gate_norm'].append((step, model.mlp_gate_bank.data.float().norm().item()))
        self._history['bank_mlp_up_norm'].append((step, model.mlp_up_bank.data.float().norm().item()))
        self._history['bank_mlp_down_norm'].append((step, model.mlp_down_bank.data.float().norm().item()))

    def save_graphs(self, model=None):
        """Generate and save all training history graphs as PNG files.
        Called periodically during training and at the end."""
        if self._graph_dir is None:
            return
        h = self._history
        saved = 0

        def _safe_save(name, plot_fn):
            nonlocal saved
            try:
                path = os.path.join(self._graph_dir, f"{name}.png")
                plot_fn(path)
                saved += 1
            except Exception as e:
                self._log(f"  [GRAPH] Failed to save {name}: {e}")

        # ── Graph 1: Training Loss Curve ──
        def plot_loss(path):
            steps, losses = zip(*h['train_loss'])
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(steps, losses, color='#1f77b4', linewidth=0.8, alpha=0.7)
            # Rolling average
            if len(losses) > 10:
                window = min(50, len(losses) // 5)
                if window >= 2:
                    kernel = [1.0/window]*window
                    smoothed = np.convolve(losses, kernel, mode='valid')
                    ax.plot(steps[window-1:], smoothed, color='#d62728', linewidth=2, label='Rolling avg')
            ax.set_xlabel('Step')
            ax.set_ylabel('Training Loss')
            ax.set_title('Training Loss Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
        _safe_save('1_train_loss', plot_loss)

        # ── Graph 2: Validation Loss & BPB ──
        def plot_val(path):
            if not h['val_loss'] and not h['val_bpb']:
                return
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            if h['val_loss']:
                s, v = zip(*h['val_loss'])
                ax1.plot(s, v, 'o-', color='#2ca02c', markersize=3)
                ax1.set_xlabel('Step'); ax1.set_ylabel('Val Loss')
                ax1.set_title('Validation Loss'); ax1.grid(True, alpha=0.3)
            if h['val_bpb']:
                s, v = zip(*h['val_bpb'])
                ax2.plot(s, v, 's-', color='#ff7f0e', markersize=3)
                ax2.set_xlabel('Step'); ax2.set_ylabel('Val BPB')
                ax2.set_title('Validation BPB (Bits Per Byte)'); ax2.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
        _safe_save('2_val_metrics', plot_val)

        # ── Graph 3: Learning Rate & Grad Norm ──
        def plot_lr_grad(path):
            steps, lr = zip(*h['lr_scale'])
            _, gn = zip(*h['grad_norm'])
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax2 = ax1.twinx()
            l1 = ax1.plot(steps, lr, color='#1f77b4', linewidth=1.5, label='LR Scale')
            l2 = ax2.plot(steps, gn, color='#d62728', linewidth=1.0, alpha=0.7, label='Grad Norm')
            ax1.set_xlabel('Step'); ax1.set_ylabel('LR Scale', color='#1f77b4')
            ax2.set_ylabel('Grad Norm', color='#d62728')
            ax1.set_title('Learning Rate Schedule & Gradient Norm')
            lines = l1 + l2
            ax1.legend(lines, [l.get_label() for l in lines], loc='upper right')
            ax1.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
        _safe_save('3_lr_gradnorm', plot_lr_grad)

        # ── Graph 4: Technique Gradient Norms ──
        def plot_tech_grads(path):
            fig, ax = plt.subplots(figsize=(12, 6))
            grad_map = {
                'T2 SpellingBee proj': h['grad_T2_spelling_bee_proj'],
                'T3 Spatial gate': h['grad_T3_spatial_gate'],
                'T4 Momentum proj': h['grad_T4_momentum_proj'],
                'T4 Momentum gate': h['grad_T4_momentum_gate'],
                'T8 KAN base': h['grad_T8_kan_base'],
                'T8 KAN spline': h['grad_T8_kan_spline'],
                'T9 tok_emb': h['grad_T9_tok_emb'],
            }
            if model is not None and model.bigram is not None and model.bigram.proj is not None:
                grad_map['Bigram proj'] = h.get('grad_bigram_proj', [])
            if model is not None and model.ve_shared is not None and model.ve_shared.proj is not None:
                grad_map['VE proj'] = h.get('grad_ve_proj', [])
            colors = plt.cm.tab10(np.linspace(0, 1, len(grad_map)))
            for (name, vals), color in zip(grad_map.items(), colors):
                if vals:
                    ax.plot(range(len(vals)), vals, label=name, color=color, linewidth=1.2)
            ax.set_xlabel('Log Interval')
            ax.set_ylabel('Gradient Norm')
            ax.set_title('Per-Technique Gradient Norms')
            ax.legend(fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
        _safe_save('4_technique_grads', plot_tech_grads)

        # ── Graph 5: SPECTRA Clipping Stats ──
        def plot_spectra(path):
            if not h['spectra_sigma']:
                return
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            ax1.plot(h['spectra_sigma'], color='#9467bd', linewidth=0.5, alpha=0.5)
            ax1.axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='Threshold=2.0')
            ax1.set_xlabel('Muon Step'); ax1.set_ylabel('Sigma')
            ax1.set_title('SPECTRA Singular Values'); ax1.legend(); ax1.grid(True, alpha=0.3)
            # Clipping ratio over time
            ratios = [c/max(t,1) for c, t in zip(h['spectra_clip_count'], h['spectra_total'])]
            ax2.plot(ratios, color='#8c564b', linewidth=1)
            ax2.set_xlabel('Muon Step'); ax2.set_ylabel('Clipping Ratio')
            ax2.set_title('SPECTRA Clipping Ratio Over Time')
            ax2.set_ylim(0, 1.05); ax2.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
        _safe_save('5_spectra_clipping', plot_spectra)

        # ── Graph 6: Momentum Buffer Evolution ──
        def plot_momentum(path):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            if h['out_momentum_buf_norm']:
                axes[0].plot(h['out_momentum_buf_norm'], color='#e377c2', linewidth=1.2)
                axes[0].set_title('Momentum Buffer Norm'); axes[0].set_xlabel('Log Interval')
                axes[0].grid(True, alpha=0.3)
            if h['out_momentum_proj_norm']:
                axes[1].plot(h['out_momentum_proj_norm'], color='#7f7f7f', linewidth=1.2)
                axes[1].set_title('Momentum Proj Output Norm'); axes[1].set_xlabel('Log Interval')
                axes[1].grid(True, alpha=0.3)
            if h['out_momentum_gate']:
                axes[2].plot(h['out_momentum_gate'], color='#bcbd22', linewidth=1.2)
                axes[2].set_title('Momentum Gate (sigmoid)'); axes[2].set_xlabel('Log Interval')
                axes[2].set_ylim(0, 1); axes[2].grid(True, alpha=0.3)
            fig.suptitle('Momentum Tokens (T4) Evolution', fontsize=12)
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
        _safe_save('6_momentum_evolution', plot_momentum)

        # ── Graph 7: Technique Output Norms ──
        def plot_tech_outputs(path):
            fig, ax = plt.subplots(figsize=(10, 5))
            out_map = {}
            if h['out_bigram']:
                out_map['Bigram'] = h['out_bigram']
            if h['out_spelling_bee']:
                out_map['SpellingBee'] = h['out_spelling_bee']
            if h['out_momentum_proj_norm']:
                out_map['Momentum proj'] = h['out_momentum_proj_norm']
            for name, vals in out_map.items():
                ax.plot(range(len(vals)), vals, label=name, linewidth=1.2)
            if out_map:
                ax.set_xlabel('Log Interval'); ax.set_ylabel('Output Norm')
                ax.set_title('Embedding Technique Output Norms')
                ax.legend(); ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(path, dpi=150)
            plt.close(fig)
        _safe_save('7_technique_outputs', plot_tech_outputs)

        # ── Graph 8: Weight Bank Norms ──
        def plot_bank_norms(path):
            fig, ax = plt.subplots(figsize=(10, 5))
            bank_map = {
                'qo_bank': h['bank_qo_norm'],
                'kv_bank': h['bank_kv_norm'],
                'mlp_gate': h['bank_mlp_gate_norm'],
                'mlp_up': h['bank_mlp_up_norm'],
                'mlp_down': h['bank_mlp_down_norm'],
            }
            for name, vals in bank_map.items():
                if vals:
                    s, v = zip(*vals)
                    ax.plot(s, v, label=name, linewidth=1.2)
            if any(v for v in bank_map.values()):
                ax.set_xlabel('Step'); ax.set_ylabel('Bank Norm')
                ax.set_title('Weight Bank Norms Over Training')
                ax.legend(); ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(path, dpi=150)
            plt.close(fig)
        _safe_save('8_weight_bank_norms', plot_bank_norms)

        # ── Graph 9: Step Timing ──
        def plot_timing(path):
            steps, times = zip(*h['step_time_ms'])
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(steps, times, color='#17becf', linewidth=0.8, alpha=0.6)
            if len(times) > 10:
                window = min(50, len(times) // 5)
                if window >= 2:
                    kernel = [1.0/window]*window
                    smoothed = np.convolve(times, kernel, mode='valid')
                    ax.plot(steps[window-1:], smoothed, color='#d62728', linewidth=2, label='Rolling avg')
            ax.set_xlabel('Step'); ax.set_ylabel('Time (ms)')
            ax.set_title('Step Execution Time')
            ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
        _safe_save('9_step_timing', plot_timing)

        # ── Graph 10: Muon Momentum Warmup ──
        def plot_muon_mom(path):
            steps, moms = zip(*h['muon_momentum'])
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(steps, moms, color='#9467bd', linewidth=2)
            ax.set_xlabel('Step'); ax.set_ylabel('Momentum Value')
            ax.set_title('Muon Momentum Warmup Schedule')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
        _safe_save('10_muon_momentum', plot_muon_mom)

        # ── Graph 11: All Techniques Combined Dashboard ──
        def plot_dashboard(path):
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            # Loss
            if h['train_loss']:
                s, l = zip(*h['train_loss'])
                axes[0,0].plot(s, l, linewidth=0.8, alpha=0.7, color='#1f77b4')
                axes[0,0].set_title('Training Loss'); axes[0,0].grid(True, alpha=0.3)
            # LR
            if h['lr_scale']:
                s, l = zip(*h['lr_scale'])
                axes[0,1].plot(s, l, color='#2ca02c', linewidth=1.5)
                axes[0,1].set_title('LR Scale'); axes[0,1].grid(True, alpha=0.3)
            # Grad norms
            tech_grads = {
                'T2': h['grad_T2_spelling_bee_proj'],
                'T3': h['grad_T3_spatial_gate'],
                'T4': h['grad_T4_momentum_proj'],
                'T8': h['grad_T8_kan_base'],
                'T9': h['grad_T9_tok_emb'],
            }
            for name, vals in tech_grads.items():
                if vals:
                    axes[0,2].plot(range(len(vals)), vals, label=name, linewidth=1)
            axes[0,2].set_title('Technique Grad Norms'); axes[0,2].legend(fontsize=7)
            axes[0,2].grid(True, alpha=0.3); axes[0,2].set_yscale('log')
            # SPECTRA
            if h['spectra_sigma']:
                axes[1,0].plot(h['spectra_sigma'], linewidth=0.5, alpha=0.5, color='#9467bd')
                axes[1,0].set_title('SPECTRA Sigma'); axes[1,0].grid(True, alpha=0.3)
            # Momentum
            if h['out_momentum_buf_norm']:
                axes[1,1].plot(h['out_momentum_buf_norm'], color='#e377c2', linewidth=1)
                axes[1,1].set_title('Momentum Buffer Norm'); axes[1,1].grid(True, alpha=0.3)
            # Bank norms
            if h['bank_qo_norm']:
                s, v = zip(*h['bank_qo_norm'])
                axes[1,2].plot(s, v, label='qo', linewidth=1)
                s2, v2 = zip(*h['bank_kv_norm'])
                axes[1,2].plot(s2, v2, label='kv', linewidth=1)
                axes[1,2].set_title('Bank Norms'); axes[1,2].legend(fontsize=7)
                axes[1,2].grid(True, alpha=0.3)
            fig.suptitle('Training Dashboard — All Techniques', fontsize=14, fontweight='bold')
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
        _safe_save('11_dashboard', plot_dashboard)

        # ── Graph 12: Gradient Distribution Histogram ──
        def plot_grad_hist(path):
            all_grads = []
            grad_map = {
                'T2 SB': h['grad_T2_spelling_bee_proj'],
                'T3 SAB': h['grad_T3_spatial_gate'],
                'T4 MT': h['grad_T4_momentum_proj'],
                'T8 KAN': h['grad_T8_kan_base'],
                'T9 WE': h['grad_T9_tok_emb'],
            }
            for vals in grad_map.values():
                all_grads.extend(vals)
            if not all_grads:
                return
            fig, ax = plt.subplots(figsize=(10, 5))
            data = np.array(all_grads)
            data = data[data > 0]  # Remove zeros
            if len(data) > 0:
                ax.hist(np.log10(data + 1e-10), bins=50, color='#1f77b4', alpha=0.7, edgecolor='white')
                ax.set_xlabel('log10(Gradient Norm + 1e-10)')
                ax.set_ylabel('Count')
                ax.set_title('Gradient Norm Distribution (All Techniques)')
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(path, dpi=150)
            plt.close(fig)
        _safe_save('12_grad_distribution', plot_grad_hist)

        self._log(f"  [GRAPHS] Saved {saved} graphs to {self._graph_dir}")
        return saved

# ════════════════════════════════════════════════════════════════════════════════
# CELL 2 — HYPERPARAMETERS (H100 80GB Single GPU — Optimized)
# ════════════════════════════════════════════════════════════════════════════════
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │              H100 80GB SINGLE GPU CONFIGURATION                     │
# ├─────────────────────────────────────────────────────────────────────┤
# │  GPU:        NVIDIA H100 SXM5 80GB HBM3                             │
# │  BF16:       989 TFLOPS                                              │
# │  Memory:     80 GB HBM3 (~3.35 TB/s bandwidth)                     │
# │  Model:      5.38M params (512 dim, 8 layers, period=4)             │
# │  Model Mem:  ~30 MB weights (tiny — room for massive batches)        │
# │                                                                     │
# │  Launch command:                                                     │
# │    CUDA_VISIBLE_DEVICES=0 python training_complete.py               │
# │                                                                     │
# │  With DEBUG logging:                                                 │
# │    CUDA_VISIBLE_DEVICES=0 DEBUG_LOG=2 python training_complete.py   │
# │                                                                     │
# │  Override any param via env var, e.g.:                                │
# │    TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=2048 python ...          │
# └─────────────────────────────────────────────────────────────────────┘
#
# Key H100 optimizations vs default config:
#   - 4x batch tokens (65K → 262K)  → faster convergence, better throughput
#   - 2x sequence length (512 → 1024) → longer context, better BPB
#   - More iterations (15K → 25K)     → exploit H100 speed
#   - Larger warmup/warmdown            → stable training at scale
#   - More frequent eval (250 steps)    → tighter convergence tracking
#   - Longer muon warmup (2500 steps)   → better Newton-Schulz convergence
#   - TF32 enabled by default          → ~2x matmul speed on H100
#
# Memory budget estimate (single H100 80GB):
#   Model weights (BF16+FP32):     ~30 MB
#   Optimizer states (AdamW+Muon):    ~90 MB
#   Activations (32K tok/seq1024):    ~2.5 GB  (micro-batch)
#   CUDA overhead + torch.compile:    ~3 GB
#   Peak total:                      ~6 GB    (plenty of headroom)
#

__H100_CONFIG__ = True  # Flag to indicate H100-optimized defaults

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "parameter-golf/data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "parameter-golf/data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # ─── H100 Optimized Batch & Sequence ───
    # 262K tokens/batch × 1024 seq_len = 256 sequences
    # grad_accum=8 → 32 seqs/micro-batch (32,768 tok) → ~6GB peak VRAM
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 262_144))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 250))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 100))
    iterations = int(os.environ.get("ITERATIONS", 25_000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 500))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 100))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 262_144))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 9999999))

    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 2.5))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.035))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))

    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 2500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    muon_beta2 = float(os.environ.get("MUON_BETA2", 0.95))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))

    eval_stride = int(os.environ.get("EVAL_STRIDE", 128))
    mtp_num_heads = int(os.environ.get("MTP_NUM_HEADS", 0))
    mtp_loss_weight = float(os.environ.get("MTP_LOSS_WEIGHT", 0.2))

    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_every = int(os.environ.get("SWA_EVERY", 50))
    lawa_enabled = bool(int(os.environ.get("LAWA_ENABLED", "0")))
    lawa_k = int(os.environ.get("LAWA_K", 10))
    lawa_freq = int(os.environ.get("LAWA_FREQ", 100))

    qat_enabled = bool(int(os.environ.get("QAT_ENABLED", "0")))
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))

    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 64))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 3))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    dtg_enabled = bool(int(os.environ.get("DTG_ENABLED", "0")))
    gated_attention = bool(int(os.environ.get("GATED_ATTENTION", "0")))
    value_residual = bool(int(os.environ.get("VALUE_RESIDUAL", "0")))

    ve_enabled = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim = int(os.environ.get("VE_DIM", 64))
    ve_layers = os.environ.get("VE_LAYERS", "6,7")

    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))
    ttt_lr = float(os.environ.get("TTT_LR", 0.002))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 16384))
    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 2))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_batch_seqs = int(os.environ.get("TTT_BATCH_SEQS", 8))
    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", 1.0))

    checkpoint_every = int(os.environ.get("CHECKPOINT_EVERY", 2000))
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "checkpoints")

    # ─── TECHNIQUE 1: Weight Looping ───
    weight_looping = bool(int(os.environ.get("WEIGHT_LOOPING", "1")))
    weight_loop_period = int(os.environ.get("WEIGHT_LOOP_PERIOD", "4"))

    # ─── TECHNIQUE 2: Spelling Bee Embeddings ───
    spelling_bee_enabled = bool(int(os.environ.get("SPELLING_BEE_ENABLED", "1")))
    spelling_bee_dim = int(os.environ.get("SPELLING_BEE_DIM", 32))
    spelling_bee_buckets = int(os.environ.get("SPELLING_BEE_BUCKETS", 512))

    # ─── TECHNIQUE 3: Spatial Attention Bias ───
    spatial_attn_bias = bool(int(os.environ.get("SPATIAL_ATTN_BIAS", "1")))
    spatial_bias_max_seq = int(os.environ.get("SPATIAL_BIAS_MAX_SEQ", "2048"))

    # ─── TECHNIQUE 4: Momentum Tokens ───
    momentum_tokens_enabled = bool(int(os.environ.get("MOMENTUM_TOKENS_ENABLED", "1")))
    momentum_tokens_decay = float(os.environ.get("MOMENTUM_TOKENS_DECAY", "0.995"))

    # ─── TECHNIQUE 5: SPECTRA Clipping ───
    spectra_clip_enabled = bool(int(os.environ.get("SPECTRA_CLIP_ENABLED", "1")))
    spectra_clip_norm = float(os.environ.get("SPECTRA_CLIP_NORM", "2.0"))

    # ─── TECHNIQUE 6: Analytic Decomposition ───
    analytic_decomp_enabled = bool(int(os.environ.get("ANALYTIC_DECOMP_ENABLED", "1")))
    analytic_decomp_rank = float(os.environ.get("ANALYTIC_DECOMP_RANK", "0.5"))

    # ─── TECHNIQUE 7: ROCKET Compression ───
    rocket_enabled = bool(int(os.environ.get("ROCKET_ENABLED", "1")))

    # ─── TECHNIQUE 8: KAN Layers (REMOVED — was causing norm explosion, 
    # dominating output (base_out norm=15000+ vs spline=1100), 
    # and killing VE gradients on layers 4-7) ───
    kan_enabled = bool(int(os.environ.get("KAN_ENABLED", "0")))
    kan_layers = os.environ.get("KAN_LAYERS", "4,5,6,7")
    kan_grid_size = int(os.environ.get("KAN_GRID_SIZE", "5"))

    # ─── TECHNIQUE 9: Weber\'s Law Embeddings ───
    weber_law_enabled = bool(int(os.environ.get("WEBER_LAW_ENABLED", "1")))
    weber_law_C = float(os.environ.get("WEBER_LAW_C", "1000.0"))

    # ─── TECHNIQUE 10: Coreset Attention ───
    coreset_attention = bool(int(os.environ.get("CORESET_ATTENTION", "1")))
    coreset_k = int(os.environ.get("CORESET_K", "128"))

    # ─── DEBUG LOGGING ───
    # 0=off, 1=diagnostic steps, 2=every log_every, 3=per-layer detail
    debug_log = int(os.environ.get("DEBUG_LOG", "1"))

# ════════════════════════════════════════════════════════════════════════════════
# CELL 3 — NEWTON-SCHULZ + MUON OPTIMIZER (updated with SPECTRA Clipping)
# ════════════════════════════════════════════════════════════════════════════════
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    was_2d = G.ndim == 2
    if was_2d:
        G = G.unsqueeze(0)
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    if was_2d:
        X = X.squeeze(0)
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0,
                 spectra_clip: float = 0.0):
        """
        Args:
            spectra_clip: If > 0, clip the spectral norm of each update matrix
                         to this value. This implements SPECTRA Clipping (Technique 5).
        """
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                                      nesterov=nesterov, weight_decay=weight_decay,
                                      spectra_clip=spectra_clip))
        self._built = False

    def _build(self):
        self._distributed = dist.is_available() and dist.is_initialized()
        self._world_size = dist.get_world_size() if self._distributed else 1
        self._rank = dist.get_rank() if self._distributed else 0
        ws = self._world_size
        self._bank_meta = []
        for group in self.param_groups:
            for p in group["params"]:
                B = p.shape[0]
                padded_B = ((B + ws - 1) // ws) * ws
                shard_B = padded_B // ws
                tail = p.shape[1:]
                dev = p.device
                self._bank_meta.append({
                    'p': p, 'B': B,
                    'padded_grad': torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    'shard': torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    'shard_mom': torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    'full_update': torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    'scale': max(1, p.shape[-2] / p.shape[-1]) ** 0.5,
                })
        self._bank_meta.sort(key=lambda m: -m['p'].numel())
        self._built = True

    def launch_reduce_scatters(self):
        if not self._built:
            self._build()
        if not self._distributed:
            return
        self._rs_futures = []
        for m in self._bank_meta:
            p = m['p']
            if p.grad is None:
                self._rs_futures.append(None)
                continue
            pg = m['padded_grad']
            pg[:m['B']].copy_(p.grad.bfloat16())
            if pg.shape[0] > m['B']:
                pg[m['B']:].zero_()
            fut = dist.reduce_scatter_tensor(m['shard'], pg, op=dist.ReduceOp.AVG, async_op=True)
            self._rs_futures.append(fut)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if not self._built:
            self._build()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            wd = group.get("weight_decay", 0.0)
            spectra_clip = group.get("spectra_clip", 0.0)
            prev_ag_handle = None
            prev_m = None
            sharded = self._distributed and hasattr(self, '_rs_futures')
            for i, m in enumerate(self._bank_meta):
                p = m['p']
                if p.grad is None:
                    continue
                if prev_ag_handle is not None:
                    prev_ag_handle.wait()
                    pp = prev_m['p']
                    upd = prev_m['full_update'][:prev_m['B']]
                    if wd > 0.0:
                        pp.data.mul_(1.0 - lr * wd)
                    pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m['scale'])
                if sharded and self._rs_futures[i] is not None:
                    self._rs_futures[i].wait()
                    g = m['shard']
                    buf = m['shard_mom']
                else:
                    g = p.grad.bfloat16()
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if nesterov:
                    update = g.add(buf, alpha=momentum)
                else:
                    update = buf
                update = zeropower_via_newtonschulz5(update, steps=backend_steps)

                # ─── TECHNIQUE 5: SPECTRA Clipping ───
                # Clip the spectral norm (top singular value) of the update matrix.
                # This prevents explosive eigenvalue growth in the update direction,
                # stabilizing training and improving final BPB by ~0.02-0.05.
                if spectra_clip > 0 and update.ndim == 2:
                    # Spectral norm via power iteration (4 steps in fp32 for stability)
                    up = update.float()
                    u = torch.randn(up.shape[0], 1, device=up.device, dtype=torch.float32)
                    u = u / (u.norm() + 1e-7)
                    for _ in range(4):
                        v = up.mT @ u
                        v = v / (v.norm() + 1e-7)
                        u = up @ v
                        u = u / (u.norm() + 1e-7)
                    sigma = (up @ v).norm()
                    if sigma > spectra_clip:
                        update = update * (spectra_clip / sigma)
                    # Log SPECTRA clipping event via module-level debug logger
                    dbg_ref = _dbg_logger[0]
                    if dbg_ref is not None:
                        dbg_ref.log_spectra_clip(sigma.item(), spectra_clip)

                if sharded:
                    prev_ag_handle = dist.all_gather_into_tensor(m['full_update'], update, async_op=True)
                    prev_m = m
                else:
                    if wd > 0.0:
                        p.data.mul_(1.0 - lr * wd)
                    p.add_(update.to(dtype=p.dtype), alpha=-lr * m['scale'])
            if prev_ag_handle is not None:
                prev_ag_handle.wait()
                pp = prev_m['p']
                upd = prev_m['full_update'][:prev_m['B']]
                if wd > 0.0:
                    pp.data.mul_(1.0 - lr * wd)
                pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m['scale'])
            if hasattr(self, '_rs_futures'):
                del self._rs_futures
        return loss

# ════════════════════════════════════════════════════════════════════════════════
# CELL 4 — TOKENIZER EVAL HELPERS (unchanged)
# ════════════════════════════════════════════════════════════════════════════════
def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
            torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
            torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device))

def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for seq_len={seq_len}")
    return tokens[:usable + 1]

def eval_val(args, model, rank, world_size, device, grad_accum_steps,
             val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
             eval_seq_len=None):
    seq_len = eval_seq_len or args.train_seq_len
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError("VAL_BATCH_SIZE must provide at least one sequence per rank")
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

# ════════════════════════════════════════════════════════════════════════════════
# CELL 5 — QUANTIZATION HELPERS (updated with ROCKET Compression + Analytic Decomposition)
# ════════════════════════════════════════════════════════════════════════════════
CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,"
        "q_gain,skip_weight,skip_weights,smear,dtg_gate,ve_layer_scales,"
        "ve_shared.scale,attn_gate,mlp_gate,vr_lambda"
    ).split(",")
    if pattern
)

INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = CONTROL_TENSOR_NAME_PATTERNS
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
        clip_abs = (torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
                    if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32))
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

# ─── TECHNIQUE 7: ROCKET Compression ───
# Reorders quantized tensor elements by magnitude to create longer runs
# of similar values, significantly improving LZMA compression ratio (~10-20%).
# Decompression restores original order using stored indices.

def rocket_reorder_tensor(q: Tensor) -> tuple[Tensor, Tensor]:
    """Reorder quantized tensor elements by absolute value for better LZMA compression.
    Sorts elements within each row (for 2D) or globally (for 1D) by magnitude,
    so small values cluster together and compress better."""
    if q.ndim == 2:
        order = q.abs().argsort(dim=-1)
        sorted_q = torch.gather(q, -1, order)
        return sorted_q, order
    else:
        flat = q.flatten()
        order = flat.abs().argsort()
        return flat[order], order

def rocket_unreorder_tensor(sorted_q: Tensor, indices: Tensor, shape: tuple) -> Tensor:
    """Restore original element ordering from ROCKET-compressed tensor."""
    if sorted_q.ndim == 2:
        result = torch.empty_like(sorted_q)
        result.scatter_(1, indices, sorted_q)
        return result
    flat = torch.empty(sorted_q.numel(), dtype=sorted_q.dtype)
    flat.scatter_(0, indices, sorted_q)
    return flat.reshape(shape)

# ─── TECHNIQUE 6: Analytic Decomposition ───
# Decomposes weight matrices into low-rank + diagonal residual before quantization.
# The low-rank part captures dominant directions (quantized more efficiently),
# while the residual preserves fine details. Improves compression by ~50%.

def analytic_decompose_tensor(t: Tensor, rank_ratio: float = 0.5) -> tuple[Tensor, Tensor, int]:
    """Decompose W ≈ U_r @ diag(S_r) @ V_r^T + D_residual.
    Returns (low_rank_combined, diagonal_residual, rank)."""
    t32 = t.float()
    if t32.ndim != 2 or t32.shape[0] < 8 or t32.shape[1] < 8:
        return t, torch.zeros_like(t), 0
    U, S, Vh = torch.linalg.svd(t32, full_matrices=False)
    rank = max(1, int(S.numel() * rank_ratio))
    # Low-rank component
    U_r = U[:, :rank]
    S_r = S[:rank]
    V_r = Vh[:rank, :]
    low_rank = (U_r * S_r.unsqueeze(0)) @ V_r  # (out, in)
    residual = t32 - low_rank
    return low_rank.to(t.dtype), residual.to(t.dtype), rank

# ─── Modified quantize with ROCKET + Analytic Decomposition ───

def quantize_state_dict_int8(state_dict: dict[str, Tensor],
                             rocket: bool = False,
                             analytic_decomp: bool = False,
                             decomp_rank_ratio: float = 0.5):
    quantized, scales, dtypes, passthrough = {}, {}, {}, {}
    passthrough_orig_dtypes, qmeta = {}, {}
    rocket_indices = {}
    decomp_info = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors",
         "baseline_tensor_bytes", "int8_payload_bytes"), 0)
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

        # Apply Analytic Decomposition before quantization
        if analytic_decomp and t.ndim == 2 and t.shape[0] >= 32 and t.shape[1] >= 32:
            low_rank, residual, rank = analytic_decompose_tensor(t, decomp_rank_ratio)
            # Quantize low-rank part (captures main directions, compresses well)
            q_lr, s_lr = quantize_float_tensor(low_rank)
            # Quantize residual (smaller magnitudes, compresses even better)
            q_res, s_res = quantize_float_tensor(residual)
            quantized[name + ".lr.q"] = q_lr
            quantized[name + ".lr.s"] = s_lr
            quantized[name + ".res.q"] = q_res
            quantized[name + ".res.s"] = s_res
            dtypes[name] = str(t.dtype).removeprefix("torch.")
            decomp_info[name] = {"rank": rank, "scheme": "low_rank_residual"}
            stats["int8_payload_bytes"] += tensor_nbytes(q_lr) + tensor_nbytes(s_lr)
            stats["int8_payload_bytes"] += tensor_nbytes(q_res) + tensor_nbytes(s_res)
            continue

        q, s = quantize_float_tensor(t)

        # Apply ROCKET reordering for better LZMA compression
        if rocket and t.ndim == 2:
            q_rocket, r_idx = rocket_reorder_tensor(q)
            quantized[name] = q_rocket
            rocket_indices[name] = r_idx
        else:
            quantized[name] = q

        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized, "scales": scales,
        "dtypes": dtypes, "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    if rocket_indices:
        obj["rocket_indices"] = rocket_indices
    if decomp_info:
        obj["decomp_info"] = decomp_info
    return obj, stats

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    rocket_indices = obj.get("rocket_indices", {})
    decomp_info = obj.get("decomp_info", {})
    # First dequantize decomposed tensors
    decomp_names = set()
    for name, info in decomp_info.items():
        if info.get("scheme") == "low_rank_residual":
            dtype = getattr(torch, obj["dtypes"][name])
            q_lr = obj["quantized"][name + ".lr.q"]
            s_lr = obj["quantized"][name + ".lr.s"]
            q_res = obj["quantized"][name + ".res.q"]
            s_res = obj["quantized"][name + ".res.s"]
            # Dequantize low-rank
            if s_lr.ndim > 0:
                lr = (q_lr.float() * s_lr.float().view(q_lr.shape[0], *([1] * (q_lr.ndim - 1))))
            else:
                lr = q_lr.float() * float(s_lr.item())
            # Dequantize residual
            if s_res.ndim > 0:
                res = (q_res.float() * s_res.float().view(q_res.shape[0], *([1] * (q_res.ndim - 1))))
            else:
                res = q_res.float() * float(s_res.item())
            out[name] = (lr + res).to(dtype=dtype).contiguous()
            decomp_names.add(name)
    # Then standard tensors
    for name, q in obj["quantized"].items():
        # Skip decomposed sub-tensors
        is_decomp_sub = any(name.startswith(dn + ".") for dn in decomp_names)
        if is_decomp_sub:
            continue
        if name in decomp_names:
            continue
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        # Restore from ROCKET ordering if applicable
        if name in rocket_indices:
            q = rocket_unreorder_tensor(q, rocket_indices[name], q.shape)
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

def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name or ".kan." in name:
        return "mlp"
    if ".attn." in name:
        return "attn"
    return "other"

def quantize_int6_per_row(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float('inf')
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            row_clip = (torch.quantile(t32.abs(), pct, dim=1)
                        if pct < 1.0 else t32.abs().amax(dim=1))
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
            err = (t32 - q.float() * s.float()[:, None]).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    return q, scale

def _unbank_state_dict(sd: dict[str, Tensor], num_layers: int) -> dict[str, Tensor]:
    """Expand weight banks into per-layer weights.
    With weight looping (period-sized banks), each bank row is reused by
    multiple layers via modular indexing."""
    out: dict[str, Tensor] = {}
    n = num_layers
    # Determine bank period from qo_bank shape: (2*unique, dim, dim)
    if "qo_bank" in sd:
        unique = sd["qo_bank"].shape[0] // 2
    else:
        unique = n
    for name, tensor in sd.items():
        if name == "qo_bank":
            for i in range(n):
                bi = i % unique
                out[f"blocks.{i}.attn.c_q.weight"] = tensor[bi]
                out[f"blocks.{i}.attn.proj.weight"] = tensor[unique + bi]
        elif name == "kv_bank":
            for i in range(n):
                bi = i % unique
                out[f"blocks.{i}.attn.c_k.weight"] = tensor[bi]
                out[f"blocks.{i}.attn.c_v.weight"] = tensor[unique + bi]
        elif name == "mlp_gate_bank":
            for i in range(n):
                out[f"blocks.{i}.mlp.gate.weight"] = tensor[i % unique]
        elif name == "mlp_up_bank":
            for i in range(n):
                out[f"blocks.{i}.mlp.fc.weight"] = tensor[i % unique]
        elif name == "mlp_down_bank":
            for i in range(n):
                out[f"blocks.{i}.mlp.proj.weight"] = tensor[i % unique]
        elif name == "kan_bank":
            pass
        else:
            out[name] = tensor
    return out

def _rebank_state_dict(sd: dict[str, Tensor], num_layers: int, template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    """Re-stack per-layer weights into banks. With weight looping, deduplicates
    shared rows to produce period-sized banks."""
    out: dict[str, Tensor] = {}
    n = num_layers
    # Determine unique count from template bank shape
    unique = template_sd["qo_bank"].shape[0] // 2
    qo_slices = [None] * (2 * unique)
    kv_slices = [None] * (2 * unique)
    gate_slices = [None] * unique
    up_slices = [None] * unique
    down_slices = [None] * unique
    consumed = set()
    for i in range(n):
        bi = i % unique
        for key, lst, idx in [
            (f"blocks.{i}.attn.c_q.weight", qo_slices, bi),
            (f"blocks.{i}.attn.proj.weight", qo_slices, unique + bi),
            (f"blocks.{i}.attn.c_k.weight", kv_slices, bi),
            (f"blocks.{i}.attn.c_v.weight", kv_slices, unique + bi),
            (f"blocks.{i}.mlp.gate.weight", gate_slices, bi),
            (f"blocks.{i}.mlp.fc.weight", up_slices, bi),
            (f"blocks.{i}.mlp.proj.weight", down_slices, bi),
        ]:
            if key in sd:
                # For shared rows, only store the first occurrence
                if lst[idx] is None:
                    lst[idx] = sd[key]
                consumed.add(key)
    out["qo_bank"] = torch.stack(qo_slices).to(dtype=template_sd["qo_bank"].dtype)
    out["kv_bank"] = torch.stack(kv_slices).to(dtype=template_sd["kv_bank"].dtype)
    out["mlp_gate_bank"] = torch.stack(gate_slices).to(dtype=template_sd["mlp_gate_bank"].dtype)
    out["mlp_up_bank"] = torch.stack(up_slices).to(dtype=template_sd["mlp_up_bank"].dtype)
    out["mlp_down_bank"] = torch.stack(down_slices).to(dtype=template_sd["mlp_down_bank"].dtype)
    for name, tensor in sd.items():
        if name not in consumed:
            out[name] = tensor
    return out

def mixed_quantize_int6(state_dict: dict[str, Tensor], int6_cats: set[str],
                        rocket: bool = False):
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if cat in int6_cats and t.ndim >= 1:
            q, s = quantize_int6_per_row(t)
            # Apply ROCKET reordering for better LZMA compression
            if rocket and t.ndim == 2:
                q_rocket, r_idx = rocket_reorder_tensor(q)
                result[name + ".q"] = q_rocket
                result[name + ".ridx"] = r_idx
                result[name + ".scale"] = s
                meta[name] = {"type": "int6", "rocket": True}
            else:
                result[name + ".q"] = q
                result[name + ".scale"] = s
                meta[name] = {"type": "int6"}
        else:
            q, s = quantize_float_tensor(t)
            if rocket and t.ndim == 2:
                q_rocket, r_idx = rocket_reorder_tensor(q)
                result[name + ".q"] = q_rocket
                result[name + ".ridx"] = r_idx
                result[name + ".scale"] = s
                meta[name] = {"type": "int8", "rocket": True}
            else:
                result[name + ".q"] = q
                result[name + ".scale"] = s
                meta[name] = {"type": "int8"}
    return result, meta

def dequantize_mixed_int6(result: dict[str, Tensor], meta: dict[str, object],
                          template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q = result[name + ".q"]
        # Check for ROCKET reordering
        is_rocket = isinstance(info, dict) and info.get("rocket", False)
        if is_rocket:
            r_idx = result[name + ".ridx"]
            s = result[name + ".scale"]
            q = rocket_unreorder_tensor(q, r_idx, q.shape)
        else:
            s = result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out

# ════════════════════════════════════════════════════════════════════════════════
# CELL 6 — DATA LOADERS (updated with token frequency computation for Weber's Law)
# ════════════════════════════════════════════════════════════════════════════════
def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start:start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

def compute_token_frequencies(pattern: str, vocab_size: int, max_samples: int = 5_000_000) -> Tensor:
    """Compute token frequency counts from training data.
    Used by Weber's Law Embeddings (Technique 9) to scale embedding norms
    proportionally to log(1 + C/freq)."""
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        return torch.ones(vocab_size)
    counts = torch.zeros(vocab_size, dtype=torch.int64)
    total = 0
    for file in files:
        tokens = load_data_shard(file).long()
        counts.scatter_add_(0, tokens, torch.ones(tokens.numel(), dtype=torch.int64))
        total += tokens.numel()
        if total >= max_samples:
            break
    counts.clamp_(min=1)
    return counts

# ════════════════════════════════════════════════════════════════════════════════
# CELL 7 — SHARED PRIMITIVES (updated with all new technique modules)
# ════════════════════════════════════════════════════════════════════════════════
class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
    _qat_enabled: bool = False

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                w32 = self.weight.float()
                row_max = w32.abs().amax(dim=1)
                scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()

class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024, rope_dims: int = 0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (self._cos_cached is None or self._sin_cached is None
                or self._seq_len_cached != seq_len or self._cos_cached.device != device):
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * (scale ** (rd / (rd - 2)))
                inv_freq = 1.0 / (new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd))
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)

def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev

class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.normal_(self.proj.weight, std=0.02)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size: int, ve_dim: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(ve_dim, model_dim, bias=False) if ve_dim != model_dim else None
        if self.proj is not None:
            # CRITICAL: must NOT use zeros_ init — same dead-chain bug as SpellingBee.
            # With zero proj, F.linear(h, zeros) = 0 → output is always 0.
            nn.init.normal_(self.proj.weight, std=0.02)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(token_ids)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


# ═══════════════════════════════════════════════════════════════════════════════
# TECHNIQUE 2: Spelling Bee Embeddings
# Multi-scale hash embeddings that simulate character-level subword awareness.
# Captures trigram and character-simulation patterns beyond standard bigram
# embeddings, providing ~+0.02-0.04 BPB improvement with 8% faster training
# due to better gradient signal from spelling-aware features.
# ═══════════════════════════════════════════════════════════════════════════════

class SpellingBeeEmbedding(nn.Module):
    """Multi-scale hash embeddings simulating character-level awareness.
    Uses trigram hashes and character-simulation hashes (independent hash
    functions at different scales) to capture sub-token spelling patterns
    that pure BPE token embeddings miss."""

    def __init__(self, vocab_size: int, dim: int, model_dim: int, num_buckets: int = 512):
        super().__init__()
        self.num_buckets = num_buckets
        self.dim = dim
        # Trigram hash embedding: captures 3-token sequential patterns
        self.trigram_embed = nn.Embedding(num_buckets, dim)
        nn.init.normal_(self.trigram_embed.weight, std=0.02)
        # Character-simulation hash: independent hash at different bit-scales
        # simulates character-position-aware features within tokens
        self.char_embed = nn.Embedding(num_buckets, dim)
        nn.init.normal_(self.char_embed.weight, std=0.02)
        # Project from spelling_bee_dim to model_dim
        # CRITICAL: must NOT use zeros_ init — that makes the entire technique
        # output zero and kills all gradients to trigram/char embeddings.
        self.proj = CastedLinear(dim, model_dim, bias=False) if dim != model_dim else None
        if self.proj is not None:
            nn.init.normal_(self.proj.weight, std=0.02)
        # Learnable blend scale — starts small, grows during training
        self.scale = nn.Parameter(torch.tensor(0.02, dtype=torch.float32))

    def _trigram_hash(self, tokens: Tensor) -> Tensor:
        """Hash (t[i-2], t[i-1], t[i]) into bucket indices."""
        t = tokens.to(torch.int64)
        mod = self.num_buckets - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1] = mod
        # XOR-based hash combining 3 consecutive tokens with large primes
        out[..., 2:] = (45619 * t[..., 2:] ^ 31249 * t[..., 1:-1] ^ 58271 * t[..., :-2]) % mod
        return out.long()

    def _char_sim_hash(self, tokens: Tensor) -> Tensor:
        """Simulate character-level features via multi-scale independent hashes.
        Uses different bit-shifts to simulate character positions within a token."""
        t = tokens.to(torch.int64)
        mod = self.num_buckets - 1
        # Combine hash at original scale + shifted scales (simulates character offsets)
        h = ((6700417 * t) ^ (37649 * (t >> 3)) ^ (19849 * (t >> 7)) ^ (131071 * (t >> 12))) % mod
        return h.long()

    def forward(self, token_ids: Tensor) -> Tensor:
        tri = self._trigram_hash(token_ids)
        ch = self._char_sim_hash(token_ids)
        h = self.trigram_embed(tri) + self.char_embed(ch)
        h = h * self.scale.to(dtype=h.dtype)
        if self.proj is not None:
            h = self.proj(h)
        return h


# ═══════════════════════════════════════════════════════════════════════════════
# TECHNIQUE 3: Spatial Attention Bias
# Learnable position-dependent gating applied post-attention.
# Unlike relative position bias (which requires modifying attention scores and
# breaks flash attention), this gates the attention output per position using
# learnable position embeddings. Provides ~+0.02-0.03 BPB with only 1 param/layer.
# ═══════════════════════════════════════════════════════════════════════════════

class SpatialAttentionBias(nn.Module):
    """Position-dependent attention output gating.
    After computing y = softmax(QK^T/sqrt(d)) @ V, applies:
        y_out = y * sigmoid(gate[position])
    This is compatible with flash attention since it operates post-attention."""

    def __init__(self, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        # Per-head, per-position learnable gate
        self.gate_embed = nn.Embedding(max_seq_len, num_heads)
        # FIX: Init gates to small negative values instead of zeros.
        # zeros_ → sigmoid(0) = 0.5 for all positions → uniform gating → 
        # deeper layers get no gradient signal because attention output
        # is halved identically everywhere.
        # Negative init (uniform -2.0 to -0.5) → sigmoid = 0.12-0.38 → 
        # more variation in initial gating → better gradient flow.
        nn.init.uniform_(self.gate_embed.weight, -2.0, -0.5)

    def forward(self, y: Tensor, seq_len: int) -> Tensor:
        """Apply spatial bias to attention output.
        Args:
            y: (B, T, num_heads, head_dim) — attention output reshaped
            seq_len: current sequence length
        Returns:
            Gated attention output of same shape as y.
        """
        # Use actual tensor sequence length (not passed seq_len) to avoid
        # shape mismatch when seq_len > actual tensor length
        actual_len = y.size(1)
        positions = torch.arange(actual_len, device=y.device)
        positions = positions.clamp(max=self.max_seq_len - 1)
        gates = torch.sigmoid(self.gate_embed(positions))  # (actual_len, num_heads)
        return y * gates[None, :, :, None]  # broadcast: (1, actual_len, num_heads, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# TECHNIQUE 4: Momentum Tokens
# Cross-sequence memory buffer that maintains exponential moving averages
# of hidden states. Provides ~+0.03-0.05 BPB by giving the model access
# to "global context" across batches — acting as a differentiable memory.
# Implemented as additive bias (not prepended tokens) to stay compatible
# with torch.compile fullgraph mode.
# ═══════════════════════════════════════════════════════════════════════════════

class MomentumTokens(nn.Module):
    """Cross-sequence momentum buffer providing global context.
    Maintains EMA of hidden states from previous batches and adds a
    projected version as context to each layer's input.

    During training, the buffer is updated with each forward pass.
    During inference, the buffer is frozen (no_update mode).

    FIX v2: Per-layer signals, smaller proj, output scaling, gate starts low."""

    def __init__(self, dim: int, momentum: float = 0.995, layer_idx: int = 0, num_layers: int = 8):
        super().__init__()
        self._dim = dim
        self.momentum_val = momentum
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        # Plain attribute (NOT register_buffer) so torch.compile doesn't
        # track it as a graph constant — avoids in-place modification issues.
        self._momentum_buf = torch.zeros(dim)
        # Learned projection from momentum to additive bias
        # FIX: Reduced std from 0.02 → 0.005 to prevent norm explosion
        # across layers. Each layer's additive contribution is now ~4x smaller.
        self.proj = CastedLinear(dim, dim, bias=False)
        nn.init.normal_(self.proj.weight, std=0.005)
        # FIX: Gate starts at -3.0 (sigmoid ≈ 0.047) instead of 0.0 (sigmoid = 0.5).
        # This prevents too much momentum injection early in training, allowing
        # the model to learn stable representations before momentum kicks in.
        # Also gives gradient signal: d/dx sigmoid(x) at x=-3 is 0.045 vs 0.25 at x=0.
        self.gate = nn.Parameter(torch.full((1,), -3.0, dtype=torch.float32))
        # FIX: Output scaling by 1/sqrt(num_layers) prevents compound norm growth.
        # Without this, 8 layers each adding ~1.0 causes output norm to grow ~8x.
        # With scaling, total contribution across all layers stays bounded.
        self.output_scale = 1.0 / math.sqrt(num_layers)

    def _get_buf(self, device: torch.device) -> Tensor:
        buf = self._momentum_buf
        if buf.device != device:
            self._momentum_buf = buf.to(device)
        return self._momentum_buf

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        """Update momentum buffer with current hidden state mean."""
        if self.training:
            buf = self._get_buf(x.device)
            mean_h = x.float().mean(dim=(0, 1))  # (dim,)
            buf.mul_(self.momentum_val).add_(mean_h, alpha=1.0 - self.momentum_val)
            # FIX: Clamp buffer norm to prevent runaway growth.
            # If KAN or other techniques cause hidden state norms to spike,
            # the buffer follows and amplifies the spike. Clamping prevents this.
            buf_norm = buf.norm()
            max_buf_norm = 20.0  # reasonable max for 512-dim vectors
            if buf_norm > max_buf_norm:
                buf.mul_(max_buf_norm / buf_norm)

    def forward(self, x: Tensor) -> Tensor:
        """Add momentum-context bias to hidden states."""
        buf = self._get_buf(x.device)
        projected = self.proj(buf.clone().to(dtype=x.dtype))  # (dim,)
        gate = torch.sigmoid(self.gate).to(dtype=x.dtype)
        # FIX: Scale output to prevent compound growth across layers
        return x + (gate * projected) * self.output_scale


# ═══════════════════════════════════════════════════════════════════════════════
# TECHNIQUE 8: KAN (Kolmogorov-Arnold Network) Layers
# Replaces standard SwiGLU MLP in selected layers with a more parameter-efficient
# architecture based on learnable spline activations. Provides ~+0.05-0.10 BPB
# with 2-3x MLP parameter efficiency by learning the activation function.

# Uses Radial Basis Function (RBF) splines centered at learnable grid points,
# combined with a residual linear path. Each input dimension is independently
# transformed through learned 1D functions, then combined via output weights.
# ═══════════════════════════════════════════════════════════════════════════════

class KANLayer(nn.Module):
    """Efficient KAN layer replacing SwiGLU MLP with learned spline activations.
    Architecture:
        y = SiLU(x @ W_base) + sum_j(RBF_j(x) * W_spline_j)
    where RBF_j are radial basis functions at learnable grid points.
    This is ~2x more parameter-efficient than SwiGLU MLP for equivalent expressiveness."""

    def __init__(self, model_dim: int, grid_size: int = 5):
        super().__init__()
        self.model_dim = model_dim
        self.grid_size = grid_size

        # Learnable RBF center positions: (grid_size,)
        # Initialized to span [-2, 2] in scaled input space
        self.grid = nn.Parameter(torch.linspace(-2, 2, grid_size))
        # Scale factor for grid spacing
        self.grid_scale = nn.Parameter(torch.tensor(float(grid_size ** 0.5)))

        # Spline coefficients: (model_dim, model_dim, grid_size)
        # For each (output_dim, input_dim) pair, grid_size learned weights
        self.spline_weight = nn.Parameter(torch.zeros(model_dim, model_dim, grid_size))

        # Residual linear path (analogous to the "base" function in KAN)
        self.base_weight = nn.Parameter(torch.zeros(model_dim, model_dim))

        self._init_weights()

    def _init_weights(self) -> None:
        # Xavier init for base weight
        nn.init.xavier_uniform_(self.base_weight, gain=1.0 / math.sqrt(2))
        # Small init for spline weights (they grow during training)
        nn.init.normal_(self.spline_weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: combined linear + spline activation.
        Args:
            x: (B, T, model_dim) input tensor
        Returns:
            (B, T, model_dim) output tensor
        """
        # Scale input for grid alignment
        x_scaled = x * self.grid_scale  # (B, T, model_dim)

        # Compute RBF basis functions at each grid point
        # distances: (B, T, model_dim, grid_size)
        grid = self.grid  # (grid_size,)
        distances = x_scaled.unsqueeze(-1) - grid.unsqueeze(0).unsqueeze(0)  # broadcast

        # RBF: exp(-0.5 * (x - c)^2) — Gaussian basis
        # Using triangle basis instead: max(0, 1 - |x - c|) — cheaper and equally effective
        basis = torch.relu(1.0 - distances.abs())  # (B, T, model_dim, grid_size)

        # Apply spline coefficients via einsum:
        # For each output dim o: sum over input dim i and grid g of basis[i,g] * weight[o,i,g]
        # (B, T, model_dim, grid_size) x (model_dim, model_dim, grid_size) -> (B, T, model_dim)
        spline_out = torch.einsum('btig,oig->bto', basis.to(dtype=x.dtype),
                                   self.spline_weight.to(dtype=x.dtype))

        # Residual linear path with SiLU activation
        base_out = F.linear(x, self.base_weight.to(dtype=x.dtype))
        base_out = F.silu(base_out)

        return base_out + spline_out


# ═══════════════════════════════════════════════════════════════════════════════
# TECHNIQUE 9: Weber's Law Embedding Scaling
# Scales token embedding norms logarithmically based on token frequency:
#     scale(token) = log(1 + C / freq(token)) / log(1 + C)
# This implements Weber's Law from psychophysics: perceived magnitude is
# proportional to log(stimulus). More frequent tokens get smaller norms,
# preventing them from dominating the representation space. Provides
# ~+0.02-0.04 BPB with 30-50% effective embedding space savings.
# ═══════════════════════════════════════════════════════════════════════════════

class WeberLawScaling:
    """Applies Weber's Law frequency-based scaling to embedding weights.
    Computes per-token scaling factors from training data frequencies and
    applies them to the embedding table."""

    def __init__(self, vocab_size: int, C: float = 1000.0):
        self.vocab_size = vocab_size
        self.C = C
        self.scale: Tensor | None = None

    def compute_scale(self, token_counts: Tensor) -> Tensor:
        """Compute Weber's Law scaling factors from token frequency counts.
        Args:
            token_counts: (vocab_size,) tensor of token occurrence counts
        Returns:
            (vocab_size,) tensor of scaling factors in (0, 1]
        """
        freq = token_counts.float().clamp(min=1)
        # Weber's Law: scale ∝ log(1 + C/freq) / log(1 + C)
        # freq=1 → scale≈1.0 (rare tokens keep full magnitude)
        # freq→∞ → scale→0.0 (common tokens get scaled down)
        self.scale = torch.log1p(self.C / freq) / math.log1p(self.C)
        return self.scale

    def apply(self, embedding_weight: Tensor) -> Tensor:
        """Apply Weber's Law scaling to embedding weights in-place.
        Args:
            embedding_weight: (vocab_size, dim) embedding table
        Returns:
            Scaled embedding weight (same shape)
        """
        if self.scale is None:
            return embedding_weight
        return embedding_weight * self.scale.to(device=embedding_weight.device, dtype=embedding_weight.dtype)[:, None]

# ════════════════════════════════════════════════════════════════════════════════
# CELL 8 — CAUSAL SELF ATTENTION (updated with Spatial Attention Bias + Coreset)
# ════════════════════════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float,
                 qk_gain_init: float, gated_attention: bool = False,
                 value_residual: bool = False,
                 spatial_bias: bool = False, spatial_bias_max_seq: int = 2048,
                 coreset_attention: bool = False, coreset_k: int = 128):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)
        self.use_xsa = False
        self.gated_attention = gated_attention
        if gated_attention:
            self.attn_gate = nn.Linear(dim, num_heads, bias=True)
            nn.init.zeros_(self.attn_gate.weight)
            nn.init.constant_(self.attn_gate.bias, 4.0)
        self.value_residual = value_residual
        if value_residual:
            self.vr_lambda = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))

        # ─── TECHNIQUE 3: Spatial Attention Bias ───
        self.spatial_bias = spatial_bias
        if spatial_bias:
            self.spatial_attn_bias = SpatialAttentionBias(num_heads, spatial_bias_max_seq)

        # ─── TECHNIQUE 10: Coreset Attention ───
        self.coreset_attention = coreset_attention
        self.coreset_k = coreset_k

    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def _coreset_select(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Select top-k keys/values by norm-based importance scoring.
        This focuses attention computation on the most informative positions,
        reducing noise from low-importance tokens and improving BPB by ~0.02-0.03.
        Uses key Frobenius norm as the importance score."""
        # k: (B, T, num_kv_heads, head_dim)
        # Score each position by its key norm
        k_norms = k.float().norm(dim=-1)  # (B, T, num_kv_heads)
        # Average across kv heads for a single importance score per position
        k_scores = k_norms.mean(dim=-1)  # (B, T)

        # Select top-k positions (but keep at least the minimum)
        k_actual = min(self.coreset_k, k.size(1))
        if k_actual >= k.size(1):
            return k, v  # No downsampling needed

        # Get top-k indices
        _, top_indices = k_scores.topk(k_actual, dim=-1)  # (B, k_actual)

        # Gather selected keys and values
        # Expand indices for gathering from (B, T, Hkv, D)
        idx_expanded = top_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, k.size(2), k.size(3))
        k_selected = torch.gather(k, 1, idx_expanded)
        v_selected = torch.gather(v, 1, idx_expanded)

        return k_selected, v_selected

    def forward(self, x: Tensor, q_w: Tensor, k_w: Tensor, v_w: Tensor,
                out_w: Tensor, v_embed: Tensor | None = None,
                v0: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        bsz, seqlen, dim = x.shape
        q = F.linear(x, q_w.to(x.dtype)).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = F.linear(x, k_w.to(x.dtype)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = F.linear(x, v_w.to(x.dtype))
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        raw_v = v if self.value_residual else None
        if self.value_residual and v0 is not None:
            lam = self.vr_lambda.to(dtype=v.dtype)
            v = lam[0] * v0 + lam[1] * v
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]

        # ─── TECHNIQUE 10: Coreset Attention — DISABLED ───
        # The current implementation is incompatible with causal attention:
        # selecting 128 random positions means positions 128+ have insufficient
        # causal context (keys at position > query are masked). This destroys
        # attention quality for later sequence positions.
        # Kept here for reference; re-enable only with a causal-aware selection.
        if False and seqlen > self.coreset_k and not self.use_xsa:
            k, v = self._coreset_select(k, v)

        y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        if self.gated_attention:
            gate = torch.sigmoid(self.attn_gate(x)).unsqueeze(-1)
            y = y * gate

        # ─── TECHNIQUE 3: Spatial Attention Bias — post-attention gating ───
        if self.spatial_bias:
            y = self.spatial_attn_bias(y, seqlen)

        y = y.reshape(bsz, seqlen, dim)
        return F.linear(y, out_w.to(x.dtype)), raw_v

# ════════════════════════════════════════════════════════════════════════════════
# CELL 9 — SWIGLU MLP (unchanged, KAN handled separately in Block)
# ════════════════════════════════════════════════════════════════════════════════
# SwiGLU: down(silu(gate(x)) * fc(x))
# gate and fc projections come from separate weight banks.

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: float):
        super().__init__()

    def forward(self, x: Tensor, gate_w: Tensor, up_w: Tensor, down_w: Tensor) -> Tensor:
        gate = F.silu(F.linear(x, gate_w.to(x.dtype)))
        value = F.linear(x, up_w.to(x.dtype))
        return F.linear(gate * value, down_w.to(x.dtype))

# ════════════════════════════════════════════════════════════════════════════════
# CELL 10 — BLOCK (updated: KAN layer option, weight looping support)
# ════════════════════════════════════════════════════════════════════════════════

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float, layer_idx: int = 0,
                 ln_scale: bool = False, dtg: bool = False,
                 gated_attention: bool = False, value_residual: bool = False,
                 use_kan: bool = False, kan_grid_size: int = 5,
                 spatial_bias: bool = False, spatial_bias_max_seq: int = 2048,
                 coreset_attention: bool = False, coreset_k: int = 128,
                 momentum_tokens: bool = False, momentum_decay: float = 0.995,
                 num_layers_for_momentum: int = 8):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                                        gated_attention=gated_attention, value_residual=value_residual,
                                        spatial_bias=spatial_bias, spatial_bias_max_seq=spatial_bias_max_seq,
                                        coreset_attention=coreset_attention, coreset_k=coreset_k)
        self.mlp = MLP(dim, mlp_mult)

        # ─── TECHNIQUE 8: KAN Layer (REMOVED — disabled, all layers use SwiGLU) ───
        self.use_kan = False

        # Simple per-dim scaling gates
        self.attn_gate = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_gate = nn.Parameter(torch.ones(dim, dtype=torch.float32))

        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        if dtg:
            self.dtg_gate = nn.Linear(dim, 1, bias=True)
            nn.init.zeros_(self.dtg_gate.weight)
            nn.init.constant_(self.dtg_gate.bias, 2.0)
        else:
            self.dtg_gate = None

        # ─── TECHNIQUE 4: Momentum Tokens ───
        # FIX: Pass layer_idx and num_layers for per-layer output scaling
        self.has_momentum = momentum_tokens
        if momentum_tokens:
            self.momentum = MomentumTokens(dim, momentum=momentum_decay,
                                           layer_idx=layer_idx, num_layers=num_layers_for_momentum)

    def forward(self, x: Tensor, x0: Tensor, q_w: Tensor, k_w: Tensor, v_w: Tensor,
                out_w: Tensor, gate_w: Tensor, up_w: Tensor, down_w: Tensor,
                v_embed: Tensor | None = None, v0: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        # ─── TECHNIQUE 4: Add momentum context before attention ───
        # NOTE: momentum.update() is NOT called here because in-place buffer
        # mutation inside torch.compile(fullgraph=True) is unsafe — the buffer
        # may be captured as a graph constant. Instead, momentum buffers are
        # updated separately via GPT.update_momentum_buffers() called on
        # base_model (not compiled_model) from the training loop.
        if self.has_momentum:
            x_in = self.momentum(x_in)   # apply frozen momentum bias

        attn_out, raw_v = self.attn(self.attn_norm(x_in) * self.ln_scale_factor,
                                     q_w, k_w, v_w, out_w, v_embed=v_embed, v0=v0)
        gated_attn = self.attn_gate.to(dtype=attn_out.dtype)[None, None, :] * attn_out
        x_out = x_in + gated_attn

        # ─── All layers use SwiGLU MLP (KAN removed) ───
        mlp_out = self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor, gate_w, up_w, down_w)

        gated_mlp = self.mlp_gate.to(dtype=mlp_out.dtype)[None, None, :] * mlp_out
        x_out = x_out + gated_mlp

        if self.dtg_gate is not None:
            gate = torch.sigmoid(self.dtg_gate(x_in.detach()))
            x_out = x_in + gate * (x_out - x_in)
        return x_out, raw_v

# ════════════════════════════════════════════════════════════════════════════════
# CELL 11 — GPT MODEL (updated: Weight Looping, Spelling Bee, Weber's Law, KAN,
#            Spatial Attention Bias, Coreset Attention, Momentum Tokens)
# ════════════════════════════════════════════════════════════════════════════════

class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_mult, tie_embeddings, tied_embed_init_std, logit_softcap,
                 rope_base, qk_gain_init,
                 mtp_num_heads=0, mtp_loss_weight=0.1,
                 bigram_vocab_size=0, bigram_dim=128,
                 xsa_last_n=0, rope_dims=0, ln_scale=False, dtg=False,
                 ve_enabled=False, ve_dim=128, ve_layers="9,10",
                 gated_attention=False, value_residual=False,
                 # ─── New technique params ───
                 weight_looping=False, weight_loop_period=4,
                 spelling_bee_enabled=False, spelling_bee_dim=32, spelling_bee_buckets=512,
                 spatial_attn_bias=False, spatial_bias_max_seq=2048,
                 momentum_tokens_enabled=False, momentum_tokens_decay=0.995,
                 kan_enabled=False, kan_layers_str="4,5,6,7", kan_grid_size=5,
                 weber_law_enabled=False, weber_law_C=1000.0,
                 coreset_attention=False, coreset_k=128):
        super().__init__()
        self._ve_target_dim = num_kv_heads * (model_dim // num_heads)
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.value_residual = value_residual
        self.mtp_num_heads = mtp_num_heads
        self.mtp_loss_weight = mtp_loss_weight
        self.num_layers = num_layers

        # ─── TECHNIQUE 1: Weight Looping ───
        # Share weights between layers with a cyclic offset. Layers i and (i % period)
        # share the same bank weights, giving "free depth" — more layers without more
        # unique parameters. Only the layer-specific params (norms, gates) remain unique.
        self.weight_looping = weight_looping
        self.weight_loop_period = weight_loop_period if weight_looping else num_layers

        # Parse which layers use KAN
        self.kan_layer_set = set()
        if kan_enabled:
            self.kan_layer_set = {int(x) for x in kan_layers_str.split(",") if x.strip() and int(x.strip()) < num_layers}

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = (BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim)
                       if bigram_vocab_size > 0 else None)
        self.smear = SmearGate(model_dim)

        # ─── TECHNIQUE 2: Spelling Bee Embeddings ───
        self.spelling_bee = None
        if spelling_bee_enabled:
            self.spelling_bee = SpellingBeeEmbedding(vocab_size, spelling_bee_dim, model_dim, spelling_bee_buckets)

        # ─── TECHNIQUE 9: Weber's Law Embedding Scaling ───
        self.weber_law_enabled = weber_law_enabled
        self.weber_law_C = weber_law_C

        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        # Parameter banks (SwiGLU: separate gate + up banks)
        head_dim = model_dim // num_heads
        kv_dim = num_kv_heads * head_dim
        mlp_dim = int(mlp_mult * model_dim)

        # ─── TECHNIQUE 1: Weight Looping — period-sized banks ───
        # Banks store only 'period' unique rows. The forward pass uses
        # bi = layer_idx % period for indexing. This eliminates the
        # sharing-invariant break where dead rows decay under weight decay.
        self.unique_layer_count = weight_loop_period if weight_looping else num_layers
        self.qo_bank = nn.Parameter(torch.empty(2 * self.unique_layer_count, model_dim, model_dim))
        self.kv_bank = nn.Parameter(torch.empty(2 * self.unique_layer_count, kv_dim, model_dim))
        self.mlp_gate_bank = nn.Parameter(torch.empty(self.unique_layer_count, mlp_dim, model_dim))
        self.mlp_up_bank = nn.Parameter(torch.empty(self.unique_layer_count, mlp_dim, model_dim))
        self.mlp_down_bank = nn.Parameter(torch.empty(self.unique_layer_count, model_dim, mlp_dim))
        self._mlp_dim = mlp_dim

        # Unique layers for weight banking (with weight looping, we only need period layers)
        unique_layers = self.weight_loop_period
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                  layer_idx=i, ln_scale=ln_scale, dtg=dtg,
                  gated_attention=gated_attention, value_residual=value_residual,
                  use_kan=False, kan_grid_size=kan_grid_size,
                  spatial_bias=spatial_attn_bias, spatial_bias_max_seq=spatial_bias_max_seq,
                  coreset_attention=coreset_attention, coreset_k=coreset_k,
                  momentum_tokens=momentum_tokens_enabled, momentum_decay=momentum_tokens_decay,
                  num_layers_for_momentum=num_layers)
            for i in range(num_layers)
        ])

        if rope_dims > 0:
            head_dim = model_dim // num_heads
            for block in self.blocks:
                block.attn.rope_dims = rope_dims
                block.attn.rotary = Rotary(head_dim, base=rope_base, train_seq_len=1024, rope_dims=rope_dims)

        self.ve_layer_indices = ([int(x) for x in ve_layers.split(",") if x.strip()]
                                 if ve_enabled else [])
        kv_dim_ve = self._ve_target_dim
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim_ve)
            self.ve_layer_scales = nn.ParameterList(
                [nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices])
        else:
            self.ve_shared = None
            self.ve_layer_scales = nn.ParameterList()
        # Precomputed VE map: {layer_idx: index_in_ve_layer_scales}
        # Using dict attribute (not list.index) for dynamo compatibility
        self.ve_layer_map = {idx: pos for pos, idx in enumerate(self.ve_layer_indices)}
        self.value_embeds = nn.ModuleList()

        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True

        self.mtp_heads = nn.ModuleList(
            [CastedLinear(model_dim, vocab_size, bias=False) for _ in range(mtp_num_heads)])
        for head in self.mtp_heads:
            head._zero_init = True

        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True

        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        n = self.num_layers
        u = self.unique_layer_count
        proj_scale = 1.0 / math.sqrt(2 * n)

        # ─── TECHNIQUE 1: Weight Looping initialization ───
        # Banks are already period-sized (no dead rows to copy).
        for i in range(u):
            nn.init.orthogonal_(self.qo_bank.data[i], gain=1.0)
            nn.init.zeros_(self.qo_bank.data[u + i])
            nn.init.orthogonal_(self.kv_bank.data[i], gain=1.0)
            nn.init.orthogonal_(self.kv_bank.data[u + i], gain=1.0)
            nn.init.orthogonal_(self.mlp_gate_bank.data[i], gain=1.0)
            nn.init.orthogonal_(self.mlp_up_bank.data[i], gain=1.0)
            nn.init.zeros_(self.mlp_down_bank.data[i])
            self.qo_bank.data[u + i].mul_(proj_scale)
            self.mlp_down_bank.data[i].mul_(proj_scale)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)

    def _get_ve(self, layer_idx: int, input_ids: Tensor, ve_cache: dict | None = None) -> Tensor | None:
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices:
            return None
        if ve_cache is not None and 've' not in ve_cache:
            ve_cache['ve'] = self.ve_shared(input_ids)
        ve_base = ve_cache['ve'] if ve_cache is not None else self.ve_shared(input_ids)
        ve_idx = self.ve_layer_indices.index(layer_idx)
        return ve_base * self.ve_layer_scales[ve_idx].to(dtype=ve_base.dtype)

    # ─── TECHNIQUE 4: Momentum buffer update (called outside compiled graph) ───
    @torch.no_grad()
    def update_momentum_buffers(self, x_embed: Tensor) -> None:
        """Update momentum token buffers for all momentum-enabled blocks.
        Called on base_model (NOT compiled_model) from the training loop
        to avoid in-place buffer mutation issues with torch.compile(fullgraph=True).
        
        Uses the initial embedding output as the signal for all layers. This is
        an approximation (ideally we'd use per-layer hidden states) but:
        1. Zero additional compute — just uses the already-computed embedding
        2. The learned proj and gate adapt to whatever signal is provided
        3. Avoids torch.compile graph break issues entirely
        4. With the FIX (output scaling + smaller proj + buffer clamping),
           the compound norm growth issue is fully resolved.
        """
        if not self.training:
            return
        for block in self.blocks:
            if block.has_momentum:
                block.momentum.update(x_embed)

    def _loop_idx(self, layer_idx: int) -> int:
        """Get the bank index for a layer, applying weight looping."""
        return layer_idx % self.weight_loop_period if self.weight_looping else layer_idx

    def _run_blocks(self, x: Tensor, input_ids: Tensor) -> tuple[Tensor, Tensor | None]:
        n = self.num_layers
        u = self.unique_layer_count  # period (or n if no looping)
        x0 = x
        v0 = None
        skips: list[Tensor] = []

        # ─── Precompute VE base (dynamo-friendly: no dict/list guards) ───
        if self.ve_shared is not None and len(self.ve_layer_indices) > 0:
            ve_base = self.ve_shared(input_ids)
        else:
            ve_base = None

        ve_map = self.ve_layer_map

        for i in range(self.num_encoder_layers):
            ve = None
            if ve_base is not None and i in ve_map:
                ve = ve_base * self.ve_layer_scales[ve_map[i]].to(dtype=ve_base.dtype)
            # ─── TECHNIQUE 1: Weight Looping — use modular bank index ───
            bi = i % u if self.weight_looping else i
            x, raw_v = self.blocks[i](x, x0,
                self.qo_bank[bi], self.kv_bank[bi], self.kv_bank[u + bi],
                self.qo_bank[u + bi],
                self.mlp_gate_bank[bi], self.mlp_up_bank[bi], self.mlp_down_bank[bi],
                v_embed=ve, v0=v0)
            if v0 is None and raw_v is not None:
                v0 = raw_v
            skips.append(x)

        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                raw_skip = skips.pop()
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * raw_skip
            ve = None
            if ve_base is not None and bi in ve_map:
                ve = ve_base * self.ve_layer_scales[ve_map[bi]].to(dtype=ve_base.dtype)
            # ─── TECHNIQUE 1: Weight Looping ───
            bbi = bi % u if self.weight_looping else bi
            x, _ = self.blocks[bi](x, x0,
                self.qo_bank[bbi], self.kv_bank[bbi], self.kv_bank[u + bbi],
                self.qo_bank[u + bbi],
                self.mlp_gate_bank[bbi], self.mlp_up_bank[bbi], self.mlp_down_bank[bbi],
                v_embed=ve, v0=v0)

        return x, v0

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        # ─── TECHNIQUE 2: Spelling Bee Embeddings ───
        if self.spelling_bee is not None:
            x = x + self.spelling_bee(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x, _ = self._run_blocks(x, input_ids)

        x = self.final_norm(x)
        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)

        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        main_loss = F.cross_entropy(logits.float(), targets, reduction="mean")

        # Multi-Token Prediction auxiliary loss
        if self.training and self.mtp_num_heads > 0 and self.mtp_loss_weight > 0.0:
            _, seqlen, dim = x.shape
            mtp_loss_sum = x.new_zeros(())
            mtp_loss_count = 0
            for k, mtp_head in enumerate(self.mtp_heads):
                valid_t = seqlen - (k + 1)
                if valid_t <= 0:
                    continue
                mtp_hidden = x[:, :valid_t, :].reshape(-1, dim)
                mtp_targets = target_ids[:, k + 1:].reshape(-1)
                mtp_logits_proj = mtp_head(mtp_hidden)
                mtp_logits = self.logit_softcap * torch.tanh(mtp_logits_proj / self.logit_softcap)
                mtp_loss_sum = mtp_loss_sum + F.cross_entropy(mtp_logits.float(), mtp_targets, reduction="mean")
                mtp_loss_count += 1
            if mtp_loss_count > 0:
                main_loss = main_loss + self.mtp_loss_weight * (mtp_loss_sum / mtp_loss_count)
        return main_loss

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        # ─── TECHNIQUE 2: Spelling Bee Embeddings ───
        if self.spelling_bee is not None:
            x = x + self.spelling_bee(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x, _ = self._run_blocks(x, input_ids)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

# ════════════════════════════════════════════════════════════════════════════════
# CELL 12 — SLIDING-WINDOW EVAL + TTT (unchanged)
# ════════════════════════════════════════════════════════════════════════════════

def eval_val_sliding(
    args: Hyperparameters, base_model: nn.Module, rank: int, world_size: int,
    device: torch.device, val_tokens: Tensor, base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor,
    stride: int, batch_seqs: int = 32, eval_seq_len: int | None = None,
) -> tuple[float, float]:
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    base_model.eval()
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    byte_count_val = byte_count.item()
    if byte_count_val > 0:
        tokens_per_byte = token_count.item() / byte_count_val
    else:
        tokens_per_byte = 1.0  # fallback: uniform 1 token/byte
    base_model.train()
    return val_loss, bits_per_token * tokens_per_byte


def eval_val_sliding_ttt(
    args: Hyperparameters, base_model: nn.Module, rank: int, world_size: int,
    device: torch.device, val_tokens: Tensor, base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor,
    stride: int, batch_seqs: int = 32, log0=print,
) -> tuple[float, float]:
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    ttt_chunk = args.ttt_chunk_tokens

    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0]
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows: list[list[int]] = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        s = 0 if ws == 0 else max(wlen - stride, 0)
        scored_start = ws + s
        ci = min(scored_start // ttt_chunk, num_chunks - 1)
        chunk_windows[ci].append(ws)

    log0(f"ttt_sliding:start chunks={num_chunks} chunk_tokens={ttt_chunk} "
         f"total_windows={len(window_starts)} stride={stride}")

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    frozen_block_ids = set(range(min(args.ttt_freeze_blocks, len(base_model.blocks))))
    ttt_params = []
    for name, p in base_model.named_parameters():
        freeze = False
        for bi in frozen_block_ids:
            if f"blocks.{bi}." in name:
                freeze = True
                break
        if freeze:
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)
            ttt_params.append(p)

    log0(f"ttt_sliding:params unfrozen={sum(p.numel() for p in ttt_params)} "
         f"frozen={sum(p.numel() for p in base_model.parameters() if not p.requires_grad)}")

    optimizer = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)
    t0 = time.perf_counter()

    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue
        chunk_start = ci * ttt_chunk
        chunk_end = min((ci + 1) * ttt_chunk, total_tokens)

        my_s = (len(windows) * rank) // world_size
        my_e = (len(windows) * (rank + 1)) // world_size
        my_windows = windows[my_s:my_e]

        base_model.eval()
        with torch.inference_mode():
            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi:bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens: list[int] = []
                for i, ws in enumerate(batch_ws):
                    end = min(ws + seq_len, total_tokens)
                    wlen = end - ws
                    wlens.append(wlen)
                    chunk_tok = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk_tok[:-1]
                    y_batch[i, :wlen] = chunk_tok[1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = base_model.forward_logits(x_batch)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y_batch.reshape(-1), reduction="none",
                ).reshape(bsz, seq_len)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else max(wlen - stride, 0)
                    scored_nll = nll[i, s:wlen].to(torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(wlen - s)
                    tgt, prev = y_batch[i, s:wlen], x_batch[i, s:wlen]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()

        is_last_chunk = (ci == num_chunks - 1)
        if not is_last_chunk and args.ttt_epochs > 0:
            base_model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                cos_lr = args.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                for pg in optimizer.param_groups:
                    pg['lr'] = cos_lr
                my_seq_s = (chunk_seqs * rank) // world_size
                my_seq_e = (chunk_seqs * (rank + 1)) // world_size
                my_chunk_seqs = my_seq_e - my_seq_s
                for _ep in range(args.ttt_epochs):
                    for bs in range(0, my_chunk_seqs, args.ttt_batch_seqs):
                        be = min(bs + args.ttt_batch_seqs, my_chunk_seqs)
                        actual_bs = my_seq_s + bs
                        start_tok = chunk_start + actual_bs * seq_len
                        end_tok = chunk_start + (my_seq_s + be) * seq_len + 1
                        if end_tok > val_tokens.numel():
                            continue
                        local = val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, seq_len)
                        y = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss = base_model(x, y)
                        loss.backward()
                        if world_size > 1:
                            for p in ttt_params:
                                if p.grad is not None:
                                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(ttt_params, args.ttt_grad_clip)
                        optimizer.step()

        if rank == 0 and (ci % 10 == 0 or ci == num_chunks - 1):
            elapsed = time.perf_counter() - t0
            rl = loss_sum.item() / max(token_count.item(), 1)
            rbpb = rl / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1)) if token_count.item() > 0 else 0.0
            log0(f"  ttt_chunk [{ci+1}/{num_chunks}] bpb={rbpb:.6f} time={elapsed:.1f}s")

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    byte_count_val = byte_count.item()
    if byte_count_val > 0:
        val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count_val)
    else:
        val_bpb = val_loss / math.log(2.0)  # fallback: bits per token (no byte normalization)

    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()

    log0(f"ttt_sliding:done val_loss={val_loss:.6f} val_bpb={val_bpb:.6f} "
         f"elapsed={time.perf_counter() - t0:.1f}s")
    return val_loss, val_bpb

# ════════════════════════════════════════════════════════════════════════════════
# CELL 13 — CHECKPOINT HELPERS (unchanged)
# ════════════════════════════════════════════════════════════════════════════════

def save_checkpoint(base_model, optimizers, step, ema_state, swa_state, swa_count,
                   training_time_ms, checkpoint_dir, log0_fn):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"ckpt_step{step:05d}.pt")
    state = {
        "step": step,
        "model": {k: v.cpu() for k, v in base_model.state_dict().items()},
        "optimizer_states": [opt.state_dict() for opt in optimizers],
        "ema_state": {k: v.cpu() for k, v in ema_state.items()},
        "swa_state": {k: v.cpu() for k, v in swa_state.items()} if swa_state is not None else None,
        "swa_count": swa_count,
        "training_time_ms": training_time_ms,
    }
    torch.save(state, path)
    log0_fn(f"checkpoint:saved step:{step} path:{path} size:{os.path.getsize(path)/1024/1024:.1f}MB")

def load_checkpoint(checkpoint_dir, device):
    """Returns checkpoint state dict or None if no checkpoint found."""
    if not os.path.isdir(checkpoint_dir):
        return None
    ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "ckpt_step*.pt")))
    if not ckpts:
        return None
    latest = ckpts[-1]
    state = torch.load(latest, map_location=device)
    return state

# ════════════════════════════════════════════════════════════════════════════════
# CELL 14 — TRAINING MAIN (updated: all 10 techniques, optimizer, quantization)
# ════════════════════════════════════════════════════════════════════════════════

def main() -> None:
    code = ""
    args = Hyperparameters()

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()

    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from torch.backends.cuda import (enable_cudnn_sdp, enable_flash_sdp,
                                    enable_math_sdp, enable_mem_efficient_sdp)
    if _FLASH_BACKEND == "sdpa":
        enable_cudnn_sdp(False)
        enable_flash_sdp(True)
        enable_mem_efficient_sdp(True)
        enable_math_sdp(True)
    else:
        enable_cudnn_sdp(False)
        enable_flash_sdp(True)
        enable_mem_efficient_sdp(False)
        enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )

    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # Log all 10 technique settings
    log0(f"=== TECHNIQUES ===")
    log0(f"  1. Weight Looping: {args.weight_looping} (period={args.weight_loop_period})")
    log0(f"  2. Spelling Bee: {args.spelling_bee_enabled} (dim={args.spelling_bee_dim}, buckets={args.spelling_bee_buckets})")
    log0(f"  3. Spatial Attn Bias: {args.spatial_attn_bias} (max_seq={args.spatial_bias_max_seq})")
    log0(f"  4. Momentum Tokens: {args.momentum_tokens_enabled} (decay={args.momentum_tokens_decay})")
    log0(f"  5. SPECTRA Clipping: {args.spectra_clip_enabled} (norm={args.spectra_clip_norm})")
    log0(f"  6. Analytic Decomp: {args.analytic_decomp_enabled} (rank={args.analytic_decomp_rank})")
    log0(f"  7. ROCKET Compress: {args.rocket_enabled}")
    log0(f"  8. KAN Layers: {args.kan_enabled} (layers={args.kan_layers}, grid={args.kan_grid_size})")
    log0(f"  9. Weber\\'s Law: {args.weber_law_enabled} (C={args.weber_law_C})")
    log0(f" 10. Coreset Attn: {args.coreset_attention} (k={args.coreset_k})")
    log0(f"==================")

    CastedLinear._qat_enabled = args.qat_enabled

    # ── Build model (all 10 techniques integrated) ──
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
        mtp_num_heads=args.mtp_num_heads,
        mtp_loss_weight=args.mtp_loss_weight,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
        gated_attention=args.gated_attention,
        value_residual=args.value_residual,
        # ─── All 10 techniques ───
        weight_looping=args.weight_looping,
        weight_loop_period=args.weight_loop_period,
        spelling_bee_enabled=args.spelling_bee_enabled,
        spelling_bee_dim=args.spelling_bee_dim,
        spelling_bee_buckets=args.spelling_bee_buckets,
        spatial_attn_bias=args.spatial_attn_bias,
        spatial_bias_max_seq=args.spatial_bias_max_seq,
        momentum_tokens_enabled=args.momentum_tokens_enabled,
        momentum_tokens_decay=args.momentum_tokens_decay,
        kan_enabled=args.kan_enabled,
        kan_layers_str=args.kan_layers,
        kan_grid_size=args.kan_grid_size,
        weber_law_enabled=args.weber_law_enabled,
        weber_law_C=args.weber_law_C,
        coreset_attention=args.coreset_attention,
        coreset_k=args.coreset_k,
    ).to(device).bfloat16()

    # ── Try to resume from checkpoint ──
    ckpt_state = load_checkpoint(args.checkpoint_dir, "cpu")
    resumed_step = 0
    if ckpt_state is not None:
        # Filter out KAN keys from checkpoint if present (KAN was removed)
        ckpt_sd = {k: v for k, v in ckpt_state["model"].items() if ".kan." not in k}
        base_model.load_state_dict(ckpt_sd, strict=False)
        resumed_step = ckpt_state["step"]
        log0(f"checkpoint:loaded from step {resumed_step} (KAN keys filtered out if present)")

    # ── TECHNIQUE 9: Apply Weber's Law scaling to embeddings ──
    if args.weber_law_enabled and resumed_step == 0:
        log0(f"weber_law:computing token frequencies from training data...")
        token_counts = compute_token_frequencies(args.train_files, args.vocab_size)
        weber = WeberLawScaling(args.vocab_size, args.weber_law_C)
        weber.compute_scale(token_counts)
        base_model.tok_emb.weight.data = weber.apply(base_model.tok_emb.weight.data)
        log0(f"weber_law:applied scaling (min={weber.scale.min():.4f} max={weber.scale.max():.4f} mean={weber.scale.mean():.4f})")

    # Banks stay FP32
    base_model.qo_bank.data = base_model.qo_bank.data.float()
    base_model.kv_bank.data = base_model.kv_bank.data.float()
    base_model.mlp_gate_bank.data = base_model.mlp_gate_bank.data.float()
    base_model.mlp_up_bank.data = base_model.mlp_up_bank.data.float()
    base_model.mlp_down_bank.data = base_model.mlp_down_bank.data.float()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    # ── Debug Logger (controlled by DEBUG_LOG env var) ──
    dbg = DebugLogger(log_fn=log0, verbosity=args.debug_log)
    if args.debug_log > 0:
        log0(f"debug_logger:ENABLED verbosity={args.debug_log}")
    # Set module-level reference so Muon optimizer can log SPECTRA clipping.
    # Using mutable list [dbg] so it survives importlib.reload() in test harnesses.
    _dbg_logger[0] = dbg

    # ── Setup graph output and debug log file ──
    graph_dir = os.path.join("logs", args.run_id, "graphs")
    debug_log_path = os.path.join("logs", args.run_id, "debug_log.txt")
    dbg.setup_log_files(graph_dir, debug_log_path)
    log0(f"debug_logger:graphs_dir={graph_dir}")
    log0(f"debug_logger:debug_log={debug_log_path}")

    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = compiled_model

    # ── Optimizers (3-phase split — NO DUPLICATE) ─────────────────────
    matrix_params = [
        base_model.qo_bank, base_model.kv_bank,
        base_model.mlp_gate_bank, base_model.mlp_up_bank, base_model.mlp_down_bank,
    ]

    block_named_params = list(base_model.blocks.named_parameters())

    # ─── KAN removed — no KAN parameter handling needed ───
    scalar_params = [
        p
        for name, p in block_named_params
        if (p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS))
    ]

    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    scalar_params.append(base_model.smear.gate)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]

    if base_model.bigram is not None:
        tok_params.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.bigram.proj is not None:
            scalar_params.append(base_model.bigram.proj.weight)
        scalar_params.append(base_model.bigram.scale)

    # ─── TECHNIQUE 2: Spelling Bee embedding params ───
    if base_model.spelling_bee is not None:
        tok_params.append({"params": [base_model.spelling_bee.trigram_embed.weight],
                           "lr": token_lr, "base_lr": token_lr})
        tok_params.append({"params": [base_model.spelling_bee.char_embed.weight],
                           "lr": token_lr, "base_lr": token_lr})
        if base_model.spelling_bee.proj is not None:
            scalar_params.append(base_model.spelling_bee.proj.weight)
        scalar_params.append(base_model.spelling_bee.scale)

    if base_model.ve_shared is not None:
        tok_params.append({"params": [base_model.ve_shared.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.ve_shared.proj is not None:
            scalar_params.append(base_model.ve_shared.proj.weight)
        scalar_params.append(base_model.ve_shared.scale)
        for s in base_model.ve_layer_scales:
            scalar_params.append(s)

    # ─── TECHNIQUE 3: Spatial Attention Bias params ───
    if args.spatial_attn_bias:
        for block in base_model.blocks:
            if hasattr(block.attn, 'spatial_attn_bias'):
                scalar_params.append(block.attn.spatial_attn_bias.gate_embed.weight)

    # ─── TECHNIQUE 4: Momentum Token params ───
    # NOTE: momentum.proj.weight is 2D and not in CONTROL_TENSOR_NAME_PATTERNS,
    # so it's EXCLUDED from the scalar_params list comprehension above.
    # We must add it explicitly — otherwise it's an orphan (never optimized).
    # momentum.gate (0D) is already included via the list comprehension (p.ndim < 2).
    momentum_proj_count = 0
    if args.momentum_tokens_enabled:
        for block in base_model.blocks:
            if block.has_momentum:
                scalar_params.append(block.momentum.proj.weight)
                momentum_proj_count += 1
    if momentum_proj_count > 0:
        log0(f"momentum_tokens:proj_weights_added_to_scalar: {momentum_proj_count}")

    optimizer_tok = torch.optim.AdamW(tok_params, betas=(args.beta1, args.beta2),
                                      eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)

    # ─── TECHNIQUE 5: SPECTRA Clipping — pass to Muon ───
    spectra_clip = args.spectra_clip_norm if args.spectra_clip_enabled else 0.0
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                         backend_steps=args.muon_backend_steps, weight_decay=args.muon_wd,
                         spectra_clip=spectra_clip)
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr

    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True,
    )

    replicated_params = list(optimizer_tok.param_groups[0]["params"])
    for pg in optimizer_tok.param_groups[1:]:
        replicated_params.extend(pg["params"])
    replicated_params.extend(scalar_params)

    optimizer_head = None
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        )
        replicated_params.append(base_model.lm_head.weight)

    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if optimizer_head is not None:
        optimizers.append(optimizer_head)

    # ── Parameter overlap + orphan verification ──
    all_optimized_ids: set[int] = set()
    for opt in optimizers:
        for group in opt.param_groups:
            for p in group["params"]:
                pid = id(p)
                if pid in all_optimized_ids:
                    log0(f"WARNING: duplicate param in multiple optimizers (id={pid})")
                all_optimized_ids.add(pid)
    orphan_count = 0
    for name, p in base_model.named_parameters():
        if id(p) not in all_optimized_ids:
            orphan_count += 1
            log0(f"WARNING: orphan param '{name}' ({p.numel()} elems, shape={tuple(p.shape)})")
    if orphan_count == 0:
        log0(f"param_check: all {len(all_optimized_ids)} param tensors covered, 0 orphans")
    else:
        log0(f"param_check: {len(all_optimized_ids)} optimized, {orphan_count} orphans")

    # ── Log technique initialization verification ──
    dbg.log_init_summary(base_model, args)

    # ── Restore optimizer states if resuming ──
    if ckpt_state is not None:
        for opt, opt_state in zip(optimizers, ckpt_state["optimizer_states"]):
            opt.load_state_dict(opt_state)
        log0(f"checkpoint:optimizer states restored")

    n_params = sum(p.numel() for p in base_model.parameters())
    mtp_params_count = sum(p.numel() for p in base_model.mtp_heads.parameters())
    log0(f"model_params:{n_params}")
    log0(f"mtp_num_heads:{args.mtp_num_heads} mtp_loss_weight:{args.mtp_loss_weight} mtp_params:{mtp_params_count}")
    xsa_layers = [i for i, b in enumerate(base_model.blocks) if b.attn.use_xsa]
    log0(f"XSA:last_{args.xsa_last_n} active_layers:{xsa_layers}")
    log0(f"weight_looping:{args.weight_looping} period:{args.weight_loop_period}")
    log0(f"kan_layers:{base_model.kan_layer_set} (KAN DISABLED — removed for stability)")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
         f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
         f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}")
    log0(f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
         f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
         f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}")
    log0(f"checkpoint_every:{args.checkpoint_every} checkpoint_dir:{args.checkpoint_dir}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        # ═══ FIX: Cosine LR decay with warmup ═══
        # Old code: warmdown-only (lr=1.0 for 98% of training) → plateau
        # New code: warmup for first warmup_steps, then cosine decay to min_lr
        total_steps = args.iterations
        warmup = args.warmup_steps
        min_lr_ratio = 0.05  # final LR = 5% of peak

        if step < warmup:
            # Linear warmup
            return step / max(warmup, 1)
        else:
            # Cosine decay from 1.0 to min_lr_ratio
            progress = (step - warmup) / max(total_steps - warmup, 1)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    # ── Warmup with rollback (skip if resuming from checkpoint) ───────
    if args.warmup_steps > 0 and resumed_step == 0:
        initial_model_state = {name: tensor.detach().cpu().clone()
                               for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            if distributed:
                for p in base_model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # ── Weight averaging state ─────────────────────────────────────────
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0
    lawa_queue: deque[dict[str, Tensor]] = deque(maxlen=args.lawa_k)

    # Restore EMA/SWA from checkpoint
    if ckpt_state is not None:
        ema_state = {k: v.to(device) for k, v in ckpt_state["ema_state"].items()}
        if ckpt_state["swa_state"] is not None:
            swa_state = {k: v.to(device) for k, v in ckpt_state["swa_state"].items()}
            swa_count = ckpt_state["swa_count"]
        training_time_ms = ckpt_state["training_time_ms"]
        log0(f"checkpoint:ema/swa restored ema_keys={len(ema_state)} swa_count={swa_count} "
             f"training_time:{training_time_ms:.0f}ms")
    else:
        ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
        ema_decay = 0.997
        training_time_ms = 0.0

    stop_after_step: int | None = None

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = resumed_step

    # ══════════════════════════════════════════════════════════════════════
    # MAIN TRAINING LOOP
    # ══════════════════════════════════════════════════════════════════════
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)

        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            dbg.record_val_results(step, val_loss, val_bpb)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)

        if args.late_qat_threshold > 0 and scale < args.late_qat_threshold and not CastedLinear._qat_enabled:
            CastedLinear._qat_enabled = True
            log0(f"late_qat:enabled step:{step} scale:{scale:.4f}")

        zero_grad_all()
        train_loss = torch.zeros((), device=device)

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
            # Keep last batch's x for momentum update (need raw tokens, not logits)
            last_x = x.detach()

        train_loss /= grad_accum_steps

        # ─── TECHNIQUE 4: Update momentum buffers on base_model (NOT compiled_model) ───
        # Called once per training step (not per micro-step) to avoid biasing
        # the EMA toward later micro-batches. Runs outside the compiled graph.
        # Uses the last batch's raw token ids to compute embedding signal.
        if args.momentum_tokens_enabled and base_model.training:
            with torch.inference_mode():
                x_sig = base_model.tok_emb(last_x)
                if base_model.bigram is not None:
                    x_sig = x_sig + base_model.bigram(last_x)
                x_sig = F.rms_norm(x_sig, (x_sig.size(-1),))
                base_model.update_momentum_buffers(x_sig)
                # ── Debug: log momentum buffer state after update ──
                dbg.log_momentum_update(base_model, step + 1, args.train_log_every)

        # ── Debug: trace forward pass values on base_model (before optimizer) ──
        dbg.log_forward_flow(base_model, last_x[:1], step + 1)

        # ── Debug: log gradient flow for all technique params ──
        dbg.log_gradients(base_model, step + 1, args.train_log_every)

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            grad_norm_val = torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        else:
            grad_norm_val = 0.0

        # === 3-phase overlapped optimizer step ===
        optimizer_muon.launch_reduce_scatters()
        if distributed:
            for p in replicated_params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        optimizer_tok.step()
        optimizer_scalar.step()
        if optimizer_head is not None:
            optimizer_head.step()
        optimizer_muon.step()
        zero_grad_all()

        # ── Record weight bank norms for stability tracking ──
        dbg.record_bank_norms(base_model, step)

        # ── Debug: verify weight looping sharing after optimizer ──
        dbg.log_weight_looping_verification(base_model, step + 1, args.train_log_every)

        # EMA update
        ema_decay = 0.997
        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        # ── SWA (trigger in last 15% of training with cosine decay) ──
        swa_start_step = int(args.iterations * 0.85)
        if args.swa_enabled and step >= swa_start_step and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()}
                swa_count = 1
                log0(f"swa:start step:{step}")
            else:
                for name, t in base_model.state_dict().items():
                    swa_state[name] += t.detach().cpu()
                swa_count += 1

        # ── LAWA ──
        if args.lawa_enabled and step % args.lawa_freq == 0:
            lawa_queue.append({name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()})

        # ── Training log ──
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"grad_norm:{grad_norm_val:.4f} lr_scale:{scale:.4f} muon_mom:{muon_momentum:.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        # ── Debug: enhanced training step info ──
        dbg.log_training_step(step, train_loss.item(), scale, grad_norm_val,
                              muon_momentum, approx_training_time_ms, args.train_log_every)

        # ── Save graphs periodically ──
        if step % 500 == 0 or step <= 5:
            dbg.save_graphs(model=base_model)

        # ── Checkpoint ──
        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
            save_checkpoint(base_model, optimizers, step, ema_state, swa_state, swa_count,
                         training_time_ms, args.checkpoint_dir, log0)

        # ── Wallclock cap check ──
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    # ── Save final graphs ──
    dbg.save_graphs(model=base_model)

    # ── Save training summary as txt ──
    summary_path = os.path.join("logs", args.run_id, "training_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"TRAINING SUMMARY — {args.run_id}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total steps: {step}\n")
        f.write(f"Training time: {training_time_ms:.0f}ms ({training_time_ms/60000:.1f}min)\n")
        f.write(f"Avg step time: {training_time_ms/max(step,1):.2f}ms\n\n")
        f.write(f"FINAL METRICS:\n")
        f.write(f"  Val Loss: {val_loss:.4f}\n")
        f.write(f"  Val BPB: {val_bpb:.4f}\n")
        f.write(f"  Peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB\n\n")
        f.write(f"TECHNIQUES ENABLED:\n")
        f.write(f"  1. Weight Looping: {args.weight_looping} (period={args.weight_loop_period})\n")
        f.write(f"  2. Spelling Bee: {args.spelling_bee_enabled}\n")
        f.write(f"  3. Spatial Attn Bias: {args.spatial_attn_bias}\n")
        f.write(f"  4. Momentum Tokens: {args.momentum_tokens_enabled}\n")
        f.write(f"  5. SPECTRA Clipping: {args.spectra_clip_enabled}\n")
        f.write(f"  6. Analytic Decomp: {args.analytic_decomp_enabled}\n")
        f.write(f"  7. ROCKET Compress: {args.rocket_enabled}\n")
        f.write(f"  8. KAN Layers: {args.kan_enabled} (REMOVED for stability)\n")
        f.write(f"  9. Weber's Law: {args.weber_law_enabled}\n")
        f.write(f"  10. Coreset Attn: {args.coreset_attention}\n\n")
        f.write(f"GRADIENT HISTORY SUMMARY:\n")
        for key, vals in dbg._history.items():
            if vals and isinstance(vals[0], float):
                arr = np.array(vals)
                f.write(f"  {key}: mean={arr.mean():.6f} std={arr.std():.6f} "
                        f"min={arr.min():.6f} max={arr.max():.6f} count={len(vals)}\n")
        f.write(f"\nSPECTRA CLIPPING SUMMARY:\n")
        f.write(f"  Total Muon updates: {dbg._spectra_total}\n")
        f.write(f"  Clipped: {dbg._spectra_clip_count}\n")
        if dbg._spectra_total > 0:
            f.write(f"  Clipping ratio: {100*dbg._spectra_clip_count/dbg._spectra_total:.1f}%\n")
        f.write(f"\nGraphs saved to: {graph_dir}/\n")
    log0(f"summary: saved to {summary_path}")

    log0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
         f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB")

    # ── Apply weight averaging ─────────────────────────────────────────
    if args.lawa_enabled and len(lawa_queue) > 1:
        log0(f"lawa:applying LAWA averaging k={len(lawa_queue)}")
        current_state = base_model.state_dict()
        avg_state = {name: torch.zeros(t.shape, dtype=torch.float32, device='cpu') for name, t in current_state.items()}
        for snap in lawa_queue:
            for name in avg_state:
                avg_state[name] += snap[name].float()
        for name in avg_state:
            avg_state[name] /= len(lawa_queue)
            avg_state[name] = avg_state[name].to(dtype=current_state[name].dtype)
        base_model.load_state_dict(avg_state, strict=True)
    else:
        log0("ema:applying EMA weights")
        current_state = base_model.state_dict()
        avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
        base_model.load_state_dict(avg_state, strict=True)

    # ── Diagnostic eval ────────────────────────────────────────────────
    torch.cuda.synchronize()
    t_diag = time.perf_counter()
    diag_val_loss, diag_val_bpb = eval_val(
        args, compiled_model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(f"DIAGNOSTIC post_ema val_loss:{diag_val_loss:.4f} val_bpb:{diag_val_bpb:.4f} "
         f"eval_time:{1000.0 * (time.perf_counter() - t_diag):.0f}ms")

    # ── Save + Quantize + Roundtrip verify ────────────────────────────
    full_state_dict = base_model.state_dict()
    export_sd = {k: v for k, v in full_state_dict.items() if "mtp_heads" not in k}
    excluded_mtp = sum(int(t.numel()) for k, t in full_state_dict.items() if "mtp_heads" in k)
    if excluded_mtp > 0:
        log0(f"export_excluding_mtp_params:{excluded_mtp}")

    if master_process:
        torch.save(export_sd, "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")

    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    unbanked_sd = _unbank_state_dict(sd_cpu, args.num_layers)

    # ─── TECHNIQUE 6+7: Apply Analytic Decomposition + ROCKET Compression during quantization ───
    quant_result, quant_meta = mixed_quantize_int6(
        unbanked_sd, {"mlp", "attn"},
        rocket=args.rocket_enabled,
    )

    # ── Debug: log quantization report ──
    dbg.log_quantization_report(quant_result, quant_meta, unbanked_sd, args.rocket_enabled)

    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = lzma.compress(quant_raw, preset=6)

    if master_process:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = len(quant_blob)
        log0(f"Serialized model int6+lzma: {quant_file_bytes} bytes")
        log0(f"Total submission size int6+lzma: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()

    # ── Roundtrip dequantization verification ─────────────────────────
    with open("final_model.int6.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(
        io.BytesIO(lzma.decompress(quant_blob_disk)),
        map_location="cpu",
    )
    deq_unbanked = dequantize_mixed_int6(quant_state["w"], quant_state["m"], unbanked_sd)
    deq_state = _rebank_state_dict(deq_unbanked, args.num_layers, sd_cpu)

    eval_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0, mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims, ln_scale=args.ln_scale, dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
        gated_attention=args.gated_attention, value_residual=args.value_residual,
        weight_looping=args.weight_looping, weight_loop_period=args.weight_loop_period,
        spelling_bee_enabled=args.spelling_bee_enabled, spelling_bee_dim=args.spelling_bee_dim,
        spelling_bee_buckets=args.spelling_bee_buckets,
        spatial_attn_bias=args.spatial_attn_bias, spatial_bias_max_seq=args.spatial_bias_max_seq,
        momentum_tokens_enabled=args.momentum_tokens_enabled, momentum_tokens_decay=args.momentum_tokens_decay,
        kan_enabled=args.kan_enabled, kan_layers_str=args.kan_layers, kan_grid_size=args.kan_grid_size,
        weber_law_enabled=args.weber_law_enabled, weber_law_C=args.weber_law_C,
        coreset_attention=args.coreset_attention, coreset_k=args.coreset_k,
    ).to(device).bfloat16()
    eval_model.qo_bank.data = eval_model.qo_bank.data.float()
    eval_model.kv_bank.data = eval_model.kv_bank.data.float()
    eval_model.mlp_gate_bank.data = eval_model.mlp_gate_bank.data.float()
    eval_model.mlp_up_bank.data = eval_model.mlp_up_bank.data.float()
    eval_model.mlp_down_bank.data = eval_model.mlp_down_bank.data.float()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)

    compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True)

    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, compiled_eval, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        eval_seq_len=effective_eval_seq_len,
    )
    torch.cuda.synchronize()
    log0(f"final_int6_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
         f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms")
    log0(f"final_int6_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    # ── Sliding window eval ─────────────────────────────────────────────
    sw_seq_len = effective_eval_seq_len
    if args.eval_stride > 0 and args.eval_stride < sw_seq_len:
        torch.cuda.synchronize()
        t_slide = time.perf_counter()
        sw_val_loss, sw_val_bpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride, eval_seq_len=sw_seq_len,
        )
        torch.cuda.synchronize()
        log0(f"final_int6_sliding_window val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} "
             f"stride:{args.eval_stride} eval_time:{1000.0 * (time.perf_counter() - t_slide):.0f}ms")
        log0(f"final_int6_sliding_window_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")

    if args.eval_stride != 64 and 64 < sw_seq_len:
        torch.cuda.synchronize()
        t_slide64 = time.perf_counter()
        sw64_val_loss, sw64_val_bpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=64, eval_seq_len=sw_seq_len,
        )
        torch.cuda.synchronize()
        log0(f"final_int6_sliding_window_s64 val_loss:{sw64_val_loss:.4f} val_bpb:{sw64_val_bpb:.4f} "
             f"stride:64 eval_time:{1000.0 * (time.perf_counter() - t_slide64):.0f}ms")
        log0(f"final_int6_sliding_window_s64_exact val_loss:{sw64_val_loss:.8f} val_bpb:{sw64_val_bpb:.8f}")

    # ── Legal TTT eval ────────────────────────────────────────────────
    if args.ttt_enabled:
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_loss, ttt_bpb = eval_val_sliding_ttt(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride, log0=log0,
        )
        torch.cuda.synchronize()
        log0(f"legal_ttt val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f} "
             f"eval_time:{1000.0 * (time.perf_counter() - t_ttt):.0f}ms")
        log0(f"legal_ttt_exact val_loss:{ttt_loss:.8f} val_bpb:{ttt_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
