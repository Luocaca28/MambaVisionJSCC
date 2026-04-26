import os
import sys
import argparse
from datetime import datetime
import warnings
import logging
import json

try:
    from tqdm.auto import tqdm  # type: ignore[import]
except Exception:  # pragma: no cover
    tqdm = None

_SUPPRESS_WARNINGS_DEFAULT = True


def _bootstrap_warning_filters() -> None:
    """
    Suppress known-noisy deprecation warnings early (before importing timm).

    This is controlled by CLI flags:
    - default: suppress
    - `--no-suppress-warnings`: show warnings
    """

    suppress = _SUPPRESS_WARNINGS_DEFAULT
    argv = sys.argv[1:]
    if "--no-suppress-warnings" in argv:
        suppress = False
    elif "--suppress-warnings" in argv:
        suppress = True

    if not suppress:
        return

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r"Importing from timm\.models\.layers is deprecated.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r"Importing from timm\.models\.registry is deprecated.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r"`torch\.cuda\.amp\..*` is deprecated\..*",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r"The epoch parameter in `scheduler\.step\(\)` was not necessary.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r"Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`.*",
    )


_bootstrap_warning_filters()

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F

from net.network import MambaVisionJSCC
from data.datasets import get_loader
from utils import *
from loss.distortion import MS_SSIM
import csv


def _load_checkpoint_state_dict(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        # common wrappers
        for key in ("state_dict", "model", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)!r}")
    # Strip DataParallel / DDP prefix if present.
    if any(str(k).startswith("module.") for k in ckpt.keys()):
        ckpt = {str(k)[len("module.") :]: v for k, v in ckpt.items()}

    return ckpt


def _infer_ckpt_bottleneck_C(state_dict: dict) -> int | None:
    # Prefer encoder head (shape [C, embed_dim])
    for k in ("encoder.head_list.weight",):
        if k in state_dict and hasattr(state_dict[k], "shape"):
            return int(state_dict[k].shape[0])
    # Fallback: search any matching key
    for key, val in state_dict.items():
        if str(key).endswith("encoder.head_list.weight") and hasattr(val, "shape"):
            return int(val.shape[0])
    # Fallback: decoder head (shape [embed_dim, C])
    for k in ("decoder.head_list.weight",):
        if k in state_dict and hasattr(state_dict[k], "shape"):
            return int(state_dict[k].shape[1])
    for key, val in state_dict.items():
        if str(key).endswith("decoder.head_list.weight") and hasattr(val, "shape"):
            return int(val.shape[1])
    return None


parser = argparse.ArgumentParser(description="MambaVisionJSCC")
parser.add_argument("--training", action="store_true", help="training or testing")
parser.add_argument(
    "--trainset",
    type=str,
    default="DIV2K",
    choices=["DIV2K"],
    help="train dataset name (DIV2K only)",
)
parser.add_argument(
    "--testset",
    type=str,
    default="kodak",
    choices=["kodak", "CLIC21", "ffhq"],
    help="specify the testset for HR models",
)
parser.add_argument(
    "--distortion-metric",
    type=str,
    default="MSE",
    choices=["MSE", "MS-SSIM"],
    help="evaluation metrics",
)
parser.add_argument(
    "--model",
    type=str,
    default="MambaVisionJSCC",
    choices=["MambaVisionJSCC"],
    help="model name (single clean MambaVisionJSCC path)",
)
parser.add_argument(
    "--channel-type",
    type=str,
    default="awgn",
    choices=["awgn", "rayleigh"],
    help="wireless channel model, awgn or rayleigh",
)
parser.add_argument(
    "--C",
    type=str,
    default="96",
    help="bottleneck dimension (single integer; RA removed)",
)
parser.add_argument(
    "--multiple-snr",
    type=str,
    default="10",
    help="comma separated SNR list, e.g., '5,10,15'",
)
parser.add_argument(
    "--pretrain-no-channel-epochs",
    type=int,
    default=0,
    help="warmup epochs that bypass the wireless channel (pass_channel=False) before training with noise (default: 0).",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=500,
    help="total training epochs (default: 500). After finishing, run one final test on Kodak24.",
)
parser.add_argument(
    "--val-freq",
    type=int,
    default=5,
    help="run validation every N epochs (default: 5). Use 0 to disable validation during training.",
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-4,
    help="base learning rate (default: 1e-4)",
)
parser.add_argument(
    "--weight-decay",
    type=float,
    default=0.0,
    help="weight decay for AdamW (default: 0.0)",
)
parser.add_argument(
    "--scheduler",
    type=str,
    default="none",
    choices=["none", "cosine"],
    help="learning rate scheduler (default: none)",
)
parser.add_argument(
    "--min-lr",
    type=float,
    default=1e-6,
    help="minimum LR for cosine scheduler (default: 1e-6)",
)
parser.add_argument(
    "--warmup-epochs",
    type=int,
    default=0,
    help="linear warmup epochs for scheduler (default: 0)",
)
parser.add_argument(
    "--save-freq",
    type=int,
    default=None,
    help="save checkpoint every N epochs (default: dataset preset, e.g. 100 for DIV2K)",
)
parser.add_argument(
    "--save-final-only",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="only save final checkpoint (and disable periodic saves)",
)
parser.add_argument(
    "--ckpt",
    type=str,
    default=None,
    help=(
        "path to a saved model state_dict (.pth) to load. "
        "Useful for test-only runs (omit `--training`) or finetuning."
    ),
)
parser.add_argument(
    "--ckpt-strict",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "strictly enforce that the keys in `--ckpt` match the current model. "
        "Disable with `--no-ckpt-strict` if you intentionally changed the architecture."
    ),
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=None,
    help="override batch size (use small value to avoid CUDA OOM)",
)
parser.add_argument(
    "--grad-accum",
    type=int,
    default=1,
    help="gradient accumulation steps (effective batch = batch_size * grad_accum)",
)
parser.add_argument(
    "--amp",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="use torch.cuda.amp mixed precision (recommended for memory)",
)
parser.add_argument(
    "--model_size",
    type=str,
    default="small",
    choices=["small", "base", "large"],
    help="MambaVision backbone size preset (affects embed_dims / depths / heads)",
)
parser.add_argument(
    "--latent-dim",
    type=int,
    default=None,
    help="override encoder/decoder last-stage embedding dim (helps when rate is small, e.g. 96)",
)
parser.add_argument(
    "--mamba-d-state",
    type=int,
    default=8,
    help="MambaVisionMixer d_state (memory size). Official MambaVision vision setting uses 8 (default: 8).",
)
parser.add_argument(
    "--mamba-d-conv",
    type=int,
    default=3,
    help="MambaVisionMixer depthwise conv kernel size d_conv. Official MambaVision vision setting uses 3 (default: 3).",
)
parser.add_argument(
    "--mamba-expand",
    type=int,
    default=1,
    help="MambaVisionMixer channel expansion ratio expand. Official MambaVision vision setting uses 1 (default: 1).",
)
parser.add_argument(
    "--drop-path-rate",
    type=float,
    default=0.0,
    help="DropPath rate for backbone blocks (linear schedule from 0->rate). Default 0 (off) for PSNR stability.",
)
parser.add_argument(
    "--layer-scale",
    type=float,
    default=0.0,
    help="LayerScale gamma init value for Mamba/Attention blocks (<=0 disables). Default 0 (off) for PSNR stability.",
)
parser.add_argument(
    "--layer-scale-conv",
    type=float,
    default=None,
    help="LayerScale gamma init value for Conv blocks (default: same as --layer-scale; <=0 disables).",
)
parser.add_argument(
    "--attn-gate",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="enable sigmoid gating after SDPA in attention blocks (default: False).",
)
parser.add_argument(
    "--attn-gate-swin",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="enable sigmoid gating after SDPA in Swin window attention (default: inherit --attn-gate).",
)
parser.add_argument(
    "--attn-gate-mv",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="enable sigmoid gating after SDPA in MambaVision Attention blocks (default: inherit --attn-gate).",
)
parser.add_argument(
    "--enc-conv-mlp-ratio",
    type=float,
    default=0.0,
    help="token-MLP ratio inside ConvTokenBlock for encoder conv stages (0 disables).",
)
parser.add_argument(
    "--dec-conv-mlp-ratio",
    type=float,
    default=0.0,
    help="token-MLP ratio inside ConvTokenBlock for decoder high-res conv stages (0 disables). Default 0 for baseline behavior.",
)
parser.add_argument(
    "--dec-refine",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="enable a full-resolution decoder refinement head (default: False).",
)
parser.add_argument(
    "--dec-refine-ch",
    type=int,
    default=32,
    help="decoder refinement head channels (default: 32).",
)
parser.add_argument(
    "--dec-refine-depth",
    type=int,
    default=2,
    help="number of residual blocks in decoder refinement head (default: 2).",
)
parser.add_argument(
    "--dec-refine-scale",
    type=float,
    default=0.1,
    help="residual scale for decoder refinement head (default: 0.1).",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="./output",
    help="output directory for metrics/plots (contains train/, test/, picture/)",
)
parser.add_argument(
    "--suppress-warnings",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="suppress noisy third-party FutureWarnings (timm/amp deprecations)",
)
parser.add_argument(
    "--ddp-find-unused-parameters",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="DDP: enable find_unused_parameters for models with conditional branches (safer, slightly slower)",
)
parser.add_argument(
    "--sync-bn",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="DDP: convert BatchNorm to SyncBatchNorm (helps small per-GPU batch, slightly slower)",
)
parser.add_argument(
    "--eval-crop-size",
    type=int,
    default=None,
    help="center-crop size for val/test to avoid OOM on full-res images (default: no crop; use --eval-full-res for tiled eval)",
)
parser.add_argument(
    "--eval-full-res",
    action="store_true",
    help="evaluate on full resolution for val/test (uses tiled inference to avoid OOM); overrides --eval-crop-size",
)
parser.add_argument(
    "--eval-direct-full-res",
    action="store_true",
    help="force single full-image forward for val/test when --eval-full-res is set (no tiling; may OOM)",
)
parser.add_argument(
    "--eval-tile-size",
    type=int,
    default=256,
    help="tile size for full-res eval when --eval-full-res is set (default: 256)",
)
parser.add_argument(
    "--eval-tile-overlap",
    type=int,
    default=0,
    help="overlap between tiles for full-res eval (default: 0). Use small overlap to reduce seams.",
)
parser.add_argument(
    "--debug-eval-shapes",
    action="store_true",
    help="print one val/test input shape (rank0 only) to verify cropping/resolution",
)
parser.add_argument(
    "--profile-model",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="print & save model parameter size and approximate FLOPs before training/testing",
)
parser.add_argument(
    "--profile-only",
    action="store_true",
    help="run --profile-model and exit (no train/test)",
)
parser.add_argument(
    "--profile-batch",
    type=int,
    default=1,
    help="dummy batch size used for profiling forward",
)
parser.add_argument(
    "--profile-warmup",
    type=int,
    default=1,
    help="warmup iterations before collecting profiler FLOPs",
)
parser.add_argument(
    "--profile-steps",
    type=int,
    default=1,
    help="profiled iterations for FLOPs collection",
)
parser.add_argument(
    "--distributed-eval",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="run validation on all ranks (DistributedSampler) instead of rank0-only; reduces waiting time but adds reduction overhead",
)
parser.add_argument(
    "--empty-cache-before-eval",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="call torch.cuda.empty_cache() before validation (can reduce OOM but may cause large stalls)",
)
parser.add_argument(
    "--empty-cache-before-test",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="call torch.cuda.empty_cache() before final test (can reduce OOM but may cause large stalls)",
)
parser.add_argument(
    "--log-cbr",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="log/save CBR metric to logs and CSV (default: True). Use --no-log-cbr to disable.",
)
args = parser.parse_args()
# Hard-coded experiment policy (not user-configurable via CLI).
# These are forced after parsing so command-line flags cannot change them.
args.eval_direct_full_res = False

# RA (rate-adaptive) has been removed: disallow comma-separated multi-rate `--C`.
_c_parts = [p.strip() for p in str(getattr(args, "C", "")).split(",") if p.strip()]
if len(_c_parts) != 1:
    raise ValueError(
        f"Invalid --C value: {args.C!r}. "
        "Rate-adaptive (RA) has been removed, so `--C` must be a single integer (e.g., 96 or 1536)."
    )

# If a checkpoint is provided for test-only, auto-infer bottleneck C to avoid silent mismatches.
# (This repo used to save checkpoints with different C such as 1536; forcing --C=96 would fail load.)
if getattr(args, "ckpt", None) and not bool(getattr(args, "training", False)):
    try:
        if os.path.isfile(args.ckpt):
            _sd = _load_checkpoint_state_dict(args.ckpt)
            _ckpt_c = _infer_ckpt_bottleneck_C(_sd)
            if _ckpt_c is not None:
                _cur_c = int(str(args.C).split(",")[0])
                if _ckpt_c != _cur_c:
                    print(
                        f"[ckpt] Warning: checkpoint bottleneck C={_ckpt_c} does not match forced C={_cur_c}. "
                        "Since C is hard-coded in main.py, this will likely fail to load."
                    )
    except Exception as _e:
        print(f"[ckpt] Warning: failed to inspect checkpoint for auto-config: {_e!r}")

def _is_distributed_env() -> bool:
    try:
        return int(os.environ.get("WORLD_SIZE", "1")) > 1
    except Exception:
        return False


def _get_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def _get_world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def _is_main_process() -> bool:
    return _get_rank() == 0


def _ddp_all_reduce_mean(value: float) -> float:
    if not dist.is_available() or not dist.is_initialized():
        return float(value)
    t = torch.tensor([float(value)], device="cuda" if torch.cuda.is_available() else "cpu")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= float(_get_world_size())
    return float(t.item())


def _setup_distributed() -> dict:
    """
    Initialize torch.distributed using torchrun env vars.
    Returns dict with rank/world_size/local_rank and distributed flag.
    """
    info = {"distributed": False, "rank": 0, "world_size": 1, "local_rank": 0}
    if not _is_distributed_env():
        return info

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available but WORLD_SIZE>1 is set.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"

    dist.init_process_group(backend=backend, init_method="env://")
    info.update(
        distributed=True,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )
    return info


def _unique_path(path: str) -> str:
    """
    Return a non-existing path by appending _1, _2, ... before extension.
    """
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while True:
        candidate = f"{base}_{i}{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1


def _format_bytes(num_bytes: int) -> str:
    b = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024.0:
            return f"{b:.2f}{unit}"
        b /= 1024.0
    return f"{b:.2f}PB"


def _profile_model_and_save(
    net: nn.Module,
    config: "Config",
    logger: logging.Logger,
    snr: int,
    rate: int,
) -> dict:
    """
    Print & save model complexity stats.
    - Params / buffers size: exact.
    - FLOPs: uses torch.profiler(with_flops=True), may undercount custom CUDA/triton kernels.
    """
    model = net.module if hasattr(net, "module") else net
    model.eval()

    total_params = sum(int(p.numel()) for p in model.parameters())
    trainable_params = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    param_bytes = sum(int(p.numel() * p.element_size()) for p in model.parameters())
    buffer_bytes = sum(int(b.numel() * b.element_size()) for b in model.buffers())

    batch = max(1, int(getattr(args, "profile_batch", 1)))
    _, c, h, w = config.image_dims
    x = torch.randn(batch, c, h, w, device=config.device)

    flops_total = None
    prof_note = "torch.profiler(with_flops=True) approximate FLOPs; custom ops may be missing."
    try:
        from torch.profiler import profile, ProfilerActivity, schedule  # type: ignore

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        warmup = max(0, int(getattr(args, "profile_warmup", 1)))
        steps = max(1, int(getattr(args, "profile_steps", 1)))
        with profile(
            activities=activities,
            schedule=schedule(wait=0, warmup=warmup, active=steps, repeat=1),
            record_shapes=False,
            profile_memory=False,
            with_flops=True,
        ) as prof:
            iters = warmup + steps
            for _ in range(iters):
                with torch.inference_mode():
                    _ = model(x, snr, rate)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                prof.step()

        total = 0
        for evt in prof.key_averages():
            fl = getattr(evt, "flops", 0) or 0
            try:
                total += int(fl)
            except Exception:
                pass
        flops_total = float(total) if total > 0 else None
    except Exception as e:  # pragma: no cover
        prof_note = f"FLOPs unavailable (profiler failed): {type(e).__name__}: {e}"

    stats = {
        "model": getattr(args, "model", None),
        "model_size": getattr(args, "model_size", None),
        "snr": int(snr),
        "rate": int(rate),
        "input_shape": [int(batch), int(c), int(h), int(w)],
        "params_total": int(total_params),
        "params_trainable": int(trainable_params),
        "param_bytes": int(param_bytes),
        "buffer_bytes": int(buffer_bytes),
        "flops_total": flops_total,
        "note": prof_note,
        "torch_version": getattr(torch, "__version__", None),
    }

    if _is_main_process():
        logger.info(
            "[profile] "
            f"params={total_params/1e6:.3f}M (trainable {trainable_params/1e6:.3f}M) | "
            f"param_size={_format_bytes(param_bytes)} | buffer_size={_format_bytes(buffer_bytes)} | "
            f"flops={('N/A' if flops_total is None else f'{flops_total/1e9:.3f} GFLOPs')} | "
            f"input={batch}x{c}x{h}x{w} | snr={snr} rate={rate}"
        )
        profile_json_path = _unique_path(os.path.join(config.output_train, "model_profile.json"))
        with open(profile_json_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        profile_txt_path = _unique_path(os.path.join(config.output_train, "model_profile.txt"))
        with open(profile_txt_path, "w", encoding="utf-8") as f:
            f.write("MambaVision_JSCC Model Profile\n")
            f.write(f"model={stats['model']} model_size={stats['model_size']}\n")
            f.write(f"snr={stats['snr']} rate={stats['rate']}\n")
            f.write(f"input_shape={stats['input_shape']}\n")
            f.write(f"params_total={stats['params_total']} ({stats['params_total']/1e6:.3f}M)\n")
            f.write(f"params_trainable={stats['params_trainable']} ({stats['params_trainable']/1e6:.3f}M)\n")
            f.write(f"param_bytes={stats['param_bytes']} ({_format_bytes(stats['param_bytes'])})\n")
            f.write(f"buffer_bytes={stats['buffer_bytes']} ({_format_bytes(stats['buffer_bytes'])})\n")
            if stats["flops_total"] is None:
                f.write("flops_total=N/A\n")
            else:
                f.write(f"flops_total={int(stats['flops_total'])} ({stats['flops_total']/1e9:.3f} GFLOPs)\n")
            f.write(f"note={stats['note']}\n")
            f.write(f"torch_version={stats['torch_version']}\n")
        logger.info(f"[profile] saved: {profile_json_path}")
        logger.info(f"[profile] saved: {profile_txt_path}")
    return stats


def _build_optimizer(model: nn.Module, config: "Config") -> optim.Optimizer:
    """
    AdamW with decoupled weight decay.
    Applies weight decay only to "weight-like" tensors (ndim >= 2).
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if param.ndim < 2 or lname.endswith(".bias") or "norm" in lname:
            no_decay.append(param)
        else:
            decay.append(param)
    param_groups = [
        {"params": decay, "lr": config.learning_rate, "weight_decay": config.weight_decay},
        {"params": no_decay, "lr": config.learning_rate, "weight_decay": 0.0},
    ]
    return optim.AdamW(param_groups, lr=config.learning_rate, weight_decay=config.weight_decay)


def _build_scheduler(optimizer: optim.Optimizer, config: "Config"):
    sched = str(getattr(config, "scheduler", "none")).lower()
    if sched in ("none", "off", "false"):
        return None
    warmup_epochs = max(0, int(getattr(config, "warmup_epochs", 0)))
    total_epochs = max(1, int(getattr(config, "tot_epoch", 1)))
    min_lr = float(getattr(config, "min_lr", 1e-6))

    if sched == "cosine":
        if warmup_epochs > 0 and total_epochs > warmup_epochs:
            warmup = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
            )
            cosine = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs - warmup_epochs, eta_min=min_lr
            )
            return optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
            )
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=min_lr)

    return None


class Config:
    """
    MambaVisionJSCC 训练 / 测试配置，基本沿用 SwinJSCC main.py。
    """

    def __init__(self):
        self.seed = 42
        self.pass_channel = True
        self.pretrain_no_channel_epochs = max(0, int(getattr(args, "pretrain_no_channel_epochs", 0)))
        self.CUDA = True
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = (
            torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.norm = False
        self.amp = bool(args.amp) and torch.cuda.is_available()
        self.grad_accum = max(1, int(args.grad_accum))
        if bool(args.eval_full_res) or args.eval_crop_size is None:
            self.eval_crop_size = None
        else:
            self.eval_crop_size = max(16, int(args.eval_crop_size))
        # Match the official MambaVision "vision" mixer defaults (lighter & commonly stable).
        self.mamba_d_state = max(1, int(getattr(args, "mamba_d_state", 8)))
        self.mamba_d_conv = max(1, int(getattr(args, "mamba_d_conv", 3)))
        self.mamba_expand = max(1, int(getattr(args, "mamba_expand", 1)))
        self.attn_gate = bool(getattr(args, "attn_gate", False))
        _gate_swin = getattr(args, "attn_gate_swin", None)
        _gate_mv = getattr(args, "attn_gate_mv", None)
        self.attn_gate_swin = self.attn_gate if _gate_swin is None else bool(_gate_swin)
        self.attn_gate_mv = self.attn_gate if _gate_mv is None else bool(_gate_mv)
        self.drop_path_rate = max(0.0, float(getattr(args, "drop_path_rate", 0.0)))
        _ls = float(getattr(args, "layer_scale", 0.0))
        self.layer_scale = _ls if _ls and _ls > 0 else None
        _ls_conv = getattr(args, "layer_scale_conv", None)
        if _ls_conv is None:
            self.layer_scale_conv = self.layer_scale
        else:
            _v = float(_ls_conv)
            self.layer_scale_conv = _v if _v and _v > 0 else None
        self.enc_conv_mlp_ratio = max(0.0, float(getattr(args, "enc_conv_mlp_ratio", 0.0)))
        self.dec_conv_mlp_ratio = max(0.0, float(getattr(args, "dec_conv_mlp_ratio", 0.0)))
        self.dec_refine = bool(getattr(args, "dec_refine", False))
        self.dec_refine_ch = max(8, int(getattr(args, "dec_refine_ch", 32)))
        self.dec_refine_depth = max(1, int(getattr(args, "dec_refine_depth", 2)))
        self.dec_refine_scale = float(getattr(args, "dec_refine_scale", 0.1))

        # logger / 路径
        self.print_step = 100
        self.plot_step = 10000
        self.filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # save everything under output_dir, and save checkpoints directly under it (.pth)
        self.workdir = os.path.abspath(args.output_dir)
        self.log = _unique_path(os.path.join(self.workdir, f"Log_{self.filename}.log"))
        self.samples = os.path.join(self.workdir, "samples")
        self.models = self.workdir
        os.makedirs(self.workdir, exist_ok=True)
        os.makedirs(self.samples, exist_ok=True)
        self.logger = None

        # output dirs for metrics/plots
        self.output_dir = self.workdir
        self.output_train = os.path.join(self.output_dir, "train")
        self.output_test = os.path.join(self.output_dir, "test")
        self.output_picture = os.path.join(self.output_dir, "picture")
        os.makedirs(self.output_train, exist_ok=True)
        os.makedirs(self.output_test, exist_ok=True)
        os.makedirs(self.output_picture, exist_ok=True)

        # run identifier for separating outputs inside output/
        self.run_id = os.path.splitext(os.path.basename(self.log))[0].replace("Log_", "")

        # training details
        self.normalize = False
        self.learning_rate = float(getattr(args, "lr", 1e-4))
        self.weight_decay = float(getattr(args, "weight_decay", 0.0))
        self.scheduler = str(getattr(args, "scheduler", "none"))
        self.min_lr = float(getattr(args, "min_lr", 1e-6))
        self.warmup_epochs = max(0, int(getattr(args, "warmup_epochs", 0)))
        self.tot_epoch = max(1, int(getattr(args, "epochs", 1000)))

        # dataset & model-specific settings
        if args.trainset != "DIV2K":
            raise ValueError("Only DIV2K is supported; set --trainset DIV2K.")
        else:  # DIV2K
            self.save_model_freq = 100
            self.image_dims = (3, 256, 256)
            base_path = "/home/LYC/lcy/Datasets/"

            # 固定数据划分：train / val / test
            self.train_data_dir = ["/home/LYC/lcy/Datasets/DIV2K/DIV2K_train_HR/"]
            self.val_data_dir = ["/home/LYC/lcy/Datasets/DIV2K/DIV2K_valid_HR/"]
            self.test_data_dir = ["/home/LYC/lcy/Datasets/DIV2K/Kodak24/"]
            self.batch_size = 16
            self.downsample = 4
            vf = int(getattr(args, "val_freq", 5))
            self.val_freq = None if vf <= 0 else vf

            # RA (rate-adaptive) has been removed: `--C` is now a single integer bottleneck dimension.
            try:
                channel_number = int(str(args.C).split(",")[0])
            except Exception as exc:
                raise ValueError(
                    f"Invalid --C value: {args.C!r}. Expected a single integer (RA removed)."
                ) from exc

            if args.model_size == "small":
                embed = [96, 128, 192, 256]
                depths = [2, 2, 6, 2]
                heads = [3, 4, 6, 8]
            elif args.model_size == "large":
                embed = [192, 256, 320, 384]
                depths = [2, 2, 18, 2]
                heads = [6, 8, 10, 12]
            else:  # base
                embed = [128, 192, 256, 320]
                depths = [2, 2, 18, 2]
                heads = [4, 6, 8, 10]

            # Optional override for the last stage dim (shrinks channel / bottleneck width).
            # NOTE: This can reduce PSNR for reconstruction tasks; for PSNR-first runs, prefer leaving it unset.
            if getattr(args, "latent_dim", None) is not None:
                latent_dim = int(args.latent_dim)
                embed = list(embed)
                embed[-1] = latent_dim

                # Make sure num_heads divides each stage dim where attention may be used.
                heads = list(heads)
                for i in range(len(heads)):
                    if embed[i] % int(heads[i]) != 0:
                        candidate = int(heads[i])
                        while candidate > 1 and (embed[i] % candidate) != 0:
                            candidate -= 1
                        heads[i] = max(1, candidate)

            self.encoder_kwargs = dict(
                img_size=(self.image_dims[1], self.image_dims[2]),
                patch_size=2,
                in_chans=3,
                embed_dims=embed,
                depths=depths,
                num_heads=heads,
                C=channel_number,
                window_size=8,
                mlp_ratio=4.0,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
                attn_gate_swin=self.attn_gate_swin,
                attn_gate_mv=self.attn_gate_mv,
                drop_path_rate=self.drop_path_rate,
                layer_scale=self.layer_scale,
                layer_scale_conv=self.layer_scale_conv,
                conv_token_mlp_ratio=self.enc_conv_mlp_ratio,
                conv_kernel_size=3,
                mamba_d_state=self.mamba_d_state,
                mamba_d_conv=self.mamba_d_conv,
                mamba_expand=self.mamba_expand,
            )
            self.decoder_kwargs = dict(
                img_size=(self.image_dims[1], self.image_dims[2]),
                embed_dims=list(reversed(embed)),
                # Scheme C: strengthen high-resolution decoder stages (stage2/3) with one extra ConvTokenBlock each.
                # This is a small compute increase focused on texture/detail restoration.
                depths=[2, 18, 3, 3],
                num_heads=list(reversed(heads)),
                C=channel_number,
                window_size=8,
                mlp_ratio=4.0,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=nn.LayerNorm,
                patch_norm=True,
                attn_gate_swin=self.attn_gate_swin,
                attn_gate_mv=self.attn_gate_mv,
                drop_path_rate=self.drop_path_rate,
                layer_scale=self.layer_scale,
                layer_scale_conv=self.layer_scale_conv,
                conv_token_mlp_ratio=self.dec_conv_mlp_ratio,
                conv_kernel_size=5,
                refine_head=self.dec_refine,
                refine_channels=self.dec_refine_ch,
                refine_depth=self.dec_refine_depth,
                refine_scale=self.dec_refine_scale,
                mamba_d_state=self.mamba_d_state,
                mamba_d_conv=self.mamba_d_conv,
                mamba_expand=self.mamba_expand,
            )

        # allow overriding checkpoint frequency / policy
        if getattr(args, "save_freq", None) is not None:
            self.save_model_freq = max(1, int(args.save_freq))
        if bool(getattr(args, "save_final_only", False)):
            self.save_model_freq = None

        # allow overriding batch size from CLI for any dataset
        if args.batch_size is not None:
            self.batch_size = int(args.batch_size)


config = Config()

CalcuSSIM = MS_SSIM(data_range=1.0, levels=4, channel=3).to(config.device)
CalcuSSIM.eval()


def train_one_epoch(net, optimizer, train_loader, epoch, logger, global_step, scaler, config):
    net.train()
    # epoch meters for saving curves
    ep_elapsed, ep_losses, ep_psnrs, ep_msssims, ep_cbrs, ep_snrs = [
        AverageMeter() for _ in range(6)
    ]

    optimizer.zero_grad(set_to_none=True)

    # DistributedSampler needs epoch for deterministic shuffle across ranks.
    if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
        train_loader.sampler.set_epoch(epoch)

    if tqdm is not None and _is_main_process():
        iterator = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Train epoch {epoch + 1}",
            leave=False,
            dynamic_ncols=True,
        )
    else:
        iterator = train_loader

    for batch_idx, batch in enumerate(iterator):
        if isinstance(batch, (list, tuple)):
            input = batch[0]
        else:
            input = batch
        input = input.cuda(non_blocking=True)

        start_time = time.time()
        global_step += 1

        with torch.amp.autocast("cuda", enabled=config.amp):
            recon_image, CBR, SNR, mse, loss_G = net(input)
            loss = loss_G
            loss_scaled = loss / config.grad_accum

        # Use DDP no_sync on accumulation steps to reduce communication overhead.
        # IMPORTANT: If the epoch ends with a remainder (len % grad_accum != 0),
        # we must sync on the last micro-batch so DDP can all-reduce the accumulated grads.
        is_last_batch = (batch_idx + 1) == len(train_loader)
        do_sync = ((batch_idx + 1) % config.grad_accum) == 0 or is_last_batch
        use_no_sync = hasattr(net, "no_sync") and (_get_world_size() > 1) and (not do_sync)
        if use_no_sync:
            with net.no_sync():
                if config.amp:
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()
        else:
            if config.amp:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

        if (batch_idx + 1) % config.grad_accum == 0:
            if config.amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        ep_elapsed.update(time.time() - start_time)
        ep_losses.update(loss.item())
        ep_cbrs.update(CBR)
        ep_snrs.update(SNR)

        with torch.no_grad():
            mse_detached = mse.detach()
            if mse_detached.item() > 0:
                psnr = 10 * (torch.log(255.0 * 255.0 / mse_detached) / np.log(10))
                psnr_val = psnr.item()

                # MS-SSIM module is TorchScripted and does not cooperate well with AMP dtypes.
                # Always compute this metric in float32.
                with torch.amp.autocast("cuda", enabled=False):
                    msssim_val = 1 - CalcuSSIM(
                        input.detach().float(),
                        recon_image.detach().clamp(0.0, 1.0).float(),
                    ).mean().item()

                ep_psnrs.update(psnr_val)
                ep_msssims.update(msssim_val)
            else:
                ep_psnrs.update(100)
                ep_msssims.update(100)

        if tqdm is not None and _is_main_process() and isinstance(iterator, tqdm):
            if (batch_idx == 0) or ((batch_idx + 1) % config.print_step == 0):
                postfix = dict(
                    loss=f"{ep_losses.avg:.3f}",
                    snr=f"{ep_snrs.avg:.1f}",
                    psnr=f"{ep_psnrs.avg:.2f}",
                    msssim=f"{ep_msssims.avg:.3f}",
                )
                if bool(getattr(args, "log_cbr", True)):
                    postfix["cbr"] = f"{ep_cbrs.avg:.4f}"
                iterator.set_postfix(**postfix)

    # flush remainder accumulation
    if len(train_loader) % config.grad_accum != 0:
        if config.amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Reduce epoch metrics across ranks for consistent logging/CSV.
    loss_avg = _ddp_all_reduce_mean(ep_losses.avg)
    snr_avg = _ddp_all_reduce_mean(ep_snrs.avg)
    psnr_avg = _ddp_all_reduce_mean(ep_psnrs.avg)
    cbr_avg = _ddp_all_reduce_mean(ep_cbrs.avg)
    msssim_avg = _ddp_all_reduce_mean(ep_msssims.avg)

    if _is_main_process():
        parts = [
            f"[train] epoch={epoch + 1}",
            f"loss={loss_avg:.4f}",
            f"snr={snr_avg:.1f}",
            f"psnr={psnr_avg:.3f}",
        ]
        if bool(getattr(args, "log_cbr", True)):
            parts.append(f"cbr={cbr_avg:.6f}")
        parts.append(f"msssim={msssim_avg:.3f}")
        logger.info(" | ".join(parts))
    return global_step, dict(
        loss=loss_avg,
        psnr=psnr_avg,
        msssim=msssim_avg,
        cbr=cbr_avg,
        snr=snr_avg,
    )


def _append_csv_row(path, header, row_dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row_dict.get(k, "") for k in header})


def _center_crop_tensor(x: torch.Tensor, crop_size: int) -> torch.Tensor:
    """
    Center-crop BCHW tensor to crop_size x crop_size.
    If H/W smaller than crop, it will return the original tensor.
    """
    if x.dim() != 4:
        return x
    _, _, h, w = x.shape
    crop = int(crop_size)
    if crop <= 0 or h < crop or w < crop:
        return x
    top = (h - crop) // 2
    left = (w - crop) // 2
    return x[:, :, top : top + crop, left : left + crop]


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Pad BCHW tensor on bottom/right with reflect padding so that H/W are multiples of `multiple`.
    Returns padded tensor and original (H,W).
    """
    if x.dim() != 4:
        return x, (0, 0)
    _, _, h, w = x.shape
    m = int(multiple)
    if m <= 1:
        return x, (h, w)
    pad_h = (m - (h % m)) % m
    pad_w = (m - (w % m)) % m
    if pad_h == 0 and pad_w == 0:
        return x, (h, w)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (h, w)


def _reconstruct_fullres_direct(
    net,
    input_image: torch.Tensor,
    snr: int,
    rate: int,
    amp_enabled: bool,
    pad_multiple: int,
):
    """
    Full-res direct eval: one forward on the (optionally padded) full image, then unpad.
    input_image: (1,3,H,W) on CUDA
    returns: recon_image (1,3,H,W), CBR, SNR_out, mse_tensor, loss_tensor
    """
    if input_image.dim() != 4 or input_image.size(0) != 1:
        raise ValueError("Direct full-res eval expects input_image shape (1,3,H,W).")

    x, (orig_h, orig_w) = _pad_to_multiple(input_image, pad_multiple)
    net_module = net.module if hasattr(net, "module") else net
    with torch.amp.autocast("cuda", enabled=amp_enabled):
        recon, CBR, SNR_out, _, _ = net(x, snr, rate)

    recon = recon.clamp(0.0, 1.0)
    recon = recon[:, :, :orig_h, :orig_w]
    # recompute metrics on original (un-padded) area
    mse = ((input_image * 255.0 - recon * 255.0) ** 2).mean()
    loss = net_module.distortion_loss.forward(input_image, recon).mean()
    return recon, float(CBR), float(SNR_out), mse, loss


def _reconstruct_fullres_tiled(
    net,
    input_image: torch.Tensor,
    snr: int,
    rate: int,
    tile_size: int,
    overlap: int,
    amp_enabled: bool,
):
    """
    Full-res eval helper: run the network on tiles and stitch back to full resolution.

    input_image: (1, 3, H, W) on CUDA
    returns: recon_image (1, 3, H, W), CBR, SNR_out
    """
    if input_image.dim() != 4 or input_image.size(0) != 1:
        raise ValueError("Tiled eval expects input_image shape (1,3,H,W).")

    tile = int(tile_size)
    ov = int(overlap)
    if tile <= 0:
        raise ValueError("--eval-tile-size must be positive.")
    if ov < 0 or ov >= tile:
        raise ValueError("--eval-tile-overlap must satisfy 0 <= overlap < tile.")

    _, _, H, W = input_image.shape
    stride = tile - ov

    # Pad to cover full image by tiles.
    pad_h = (stride - (H - tile) % stride) % stride if H > tile else (tile - H)
    pad_w = (stride - (W - tile) % stride) % stride if W > tile else (tile - W)
    pad_h = int(pad_h)
    pad_w = int(pad_w)

    x = input_image
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, Hp, Wp = x.shape

    device = x.device
    acc = torch.zeros((1, 3, Hp, Wp), device=device, dtype=torch.float32)
    wgt = torch.zeros((1, 1, Hp, Wp), device=device, dtype=torch.float32)

    # Use a uniform weight mask; overlap will be averaged.
    weight_patch = torch.ones((1, 1, tile, tile), device=device, dtype=torch.float32)

    cbr_val = None
    snr_out_val = None
    for top in range(0, Hp - tile + 1, stride):
        for left in range(0, Wp - tile + 1, stride):
            patch = x[:, :, top : top + tile, left : left + tile]
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                recon_patch, CBR, SNR_out, _, _ = net(patch, snr, rate)
            if cbr_val is None:
                cbr_val = float(CBR)
            if snr_out_val is None:
                snr_out_val = float(SNR_out)

            recon_patch = recon_patch.clamp(0.0, 1.0).to(dtype=torch.float32)
            acc[:, :, top : top + tile, left : left + tile] += recon_patch * weight_patch
            wgt[:, :, top : top + tile, left : left + tile] += weight_patch

    recon = acc / wgt.clamp_min(1e-6)
    recon = recon[:, :, :H, :W]
    if cbr_val is None:
        cbr_val = 0.0
    if snr_out_val is None:
        snr_out_val = float(snr)
    return recon, cbr_val, snr_out_val


def evaluate_fixed(net, loader, logger, snr, rate, split_name="val"):
    distributed_eval = bool(getattr(args, "distributed_eval", False)) and _get_world_size() > 1

    # If not distributed eval, only rank0 runs to save work.
    if not distributed_eval and _get_world_size() > 1 and not _is_main_process():
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return None

    # Optionally rebuild loader with DistributedSampler for eval so各 rank分摊验证集.
    if distributed_eval:
        sampler = torch.utils.data.distributed.DistributedSampler(
            loader.dataset, shuffle=False, drop_last=False
        )
        loader = torch.utils.data.DataLoader(
            dataset=loader.dataset,
            batch_size=loader.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=loader.num_workers,
            pin_memory=getattr(loader, "pin_memory", True),
            drop_last=False,
        )

    net.eval()
    sum_loss = sum_psnr = sum_msssim = sum_cbr = sum_snr = 0.0
    count = 0
    elapsed = AverageMeter()

    with torch.inference_mode():
        if tqdm is not None and _is_main_process():
            iterator = tqdm(
                loader,
                total=len(loader),
                desc=f"{split_name} snr{snr} rate{rate}",
                leave=False,
                dynamic_ncols=True,
            )
        else:
            iterator = loader

        for batch_idx, batch in enumerate(iterator):
            if isinstance(batch, (list, tuple)):
                input = batch[0]
            else:
                input = batch
            input = input.cuda(non_blocking=True)
            if getattr(config, "eval_crop_size", None) is not None:
                input = _center_crop_tensor(input, int(config.eval_crop_size))
            if args.debug_eval_shapes and batch_idx == 0 and _is_main_process():
                logger.info(f"[{split_name}] input shape: {tuple(input.shape)}")
            start_time = time.time()

            if getattr(config, "eval_crop_size", None) is None and bool(args.eval_full_res):
                if bool(args.eval_direct_full_res):
                    # Full pipeline requires H/W divisible by the total downsample factor (PatchMerging is /2 each stage),
                    # so we pad H/W to a multiple of 2**downsample.
                    down = int(getattr(config, "downsample", 4))
                    pad_multiple = int(2**down)
                    recon_image, CBR, SNR_out, mse, loss_G = _reconstruct_fullres_direct(
                        net,
                        input,
                        snr=int(snr),
                        rate=int(rate),
                        amp_enabled=bool(config.amp),
                        pad_multiple=pad_multiple,
                    )
                else:
                    recon_image, CBR, SNR_out = _reconstruct_fullres_tiled(
                        net,
                        input,
                        snr=int(snr),
                        rate=int(rate),
                        tile_size=int(args.eval_tile_size),
                        overlap=int(args.eval_tile_overlap),
                        amp_enabled=bool(config.amp),
                    )
                    # Compute full-image metrics.
                    mse = ((input * 255.0 - recon_image * 255.0) ** 2).mean()
                    net_module = net.module if hasattr(net, "module") else net
                    loss_G = net_module.distortion_loss.forward(input, recon_image).mean()
            else:
                with torch.amp.autocast("cuda", enabled=config.amp):
                    recon_image, CBR, SNR_out, mse, loss_G = net(input, snr, rate)

            bs = input.shape[0]
            elapsed.update(time.time() - start_time)
            sum_loss += float(loss_G.item()) * bs
            sum_cbr += float(CBR) * bs
            sum_snr += float(SNR_out) * bs
            count += bs

            if mse.item() > 0:
                psnr = 10 * (torch.log(255.0 * 255.0 / mse) / np.log(10))
                sum_psnr += psnr.item() * bs
                with torch.amp.autocast("cuda", enabled=False):
                    msssim = 1 - CalcuSSIM(
                        input.float(),
                        recon_image.clamp(0.0, 1.0).float(),
                    ).mean().item()
                sum_msssim += msssim * bs
            else:
                sum_psnr += 100.0 * bs
                sum_msssim += 100.0 * bs

            if tqdm is not None and _is_main_process() and isinstance(iterator, tqdm):
                if (batch_idx == 0) or ((batch_idx + 1) % config.print_step == 0):
                    cur_count = max(1, count)
                    postfix = dict(
                        loss=f"{(sum_loss / cur_count):.3f}",
                        psnr=f"{(sum_psnr / cur_count):.2f}",
                        msssim=f"{(sum_msssim / cur_count):.3f}",
                    )
                    if bool(getattr(args, "log_cbr", True)):
                        postfix["cbr"] = f"{(sum_cbr / cur_count):.4f}"
                    iterator.set_postfix(**postfix)

    # reduce across ranks when distributed_eval=True
    if distributed_eval and dist.is_available() and dist.is_initialized():
        t = torch.tensor(
            [sum_loss, sum_psnr, sum_msssim, sum_cbr, sum_snr, float(count)],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        sum_loss, sum_psnr, sum_msssim, sum_cbr, sum_snr, count = t.tolist()

    count = max(1.0, float(count))
    loss_avg = sum_loss / count
    psnr_avg = sum_psnr / count
    msssim_avg = sum_msssim / count
    cbr_avg = sum_cbr / count
    snr_avg = sum_snr / count

    parts = [
        f"[{split_name}] snr={snr}",
        f"rate={rate}",
        f"loss={loss_avg:.4f}",
        f"psnr={psnr_avg:.3f}",
    ]
    if bool(getattr(args, "log_cbr", True)):
        parts.append(f"cbr={cbr_avg:.6f}")
    parts.append(f"msssim={msssim_avg:.3f}")
    logger.info(" | ".join(parts))
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return dict(loss=loss_avg, psnr=psnr_avg, msssim=msssim_avg, cbr=cbr_avg, snr=snr_avg)


def test(
    net,
    test_loader,
    logger,
    save_images=True,
    split_name="test",
    curve_path=None,
):
    # Only run evaluation on rank0 to avoid duplicated work & OOM.
    if _get_world_size() > 1 and not _is_main_process():
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return

    net.eval()
    if save_images and getattr(config, "eval_crop_size", None) is not None and _is_main_process():
        logger.info(
            f"[{split_name}] NOTE: saving images uses CenterCrop={int(config.eval_crop_size)}. "
            "If you want reconstructed images with the same resolution as the original files, "
            "run with `--eval-full-res` (and optionally `--eval-direct-full-res`)."
        )
    elapsed, losses, psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(6)]
    metrics = [elapsed, losses, psnrs, msssims, snrs, cbrs]

    multiple_snr = [int(x) for x in args.multiple_snr.split(",")]
    channel_number = [int(x) for x in args.C.split(",")]
    results_snr = np.zeros((len(multiple_snr), len(channel_number)))
    results_cbr = np.zeros((len(multiple_snr), len(channel_number)))
    results_psnr = np.zeros((len(multiple_snr), len(channel_number)))
    results_msssim = np.zeros((len(multiple_snr), len(channel_number)))
    results_loss = np.zeros((len(multiple_snr), len(channel_number)))

    with torch.inference_mode():
        for i, SNR in enumerate(multiple_snr):
            for j, rate in enumerate(channel_number):
                if tqdm is not None and _is_main_process():
                    iterator = tqdm(
                        test_loader,
                        total=len(test_loader),
                        desc=f"{split_name} snr{SNR} rate{rate}",
                        leave=False,
                        dynamic_ncols=True,
                    )
                else:
                    iterator = test_loader

                for batch_idx, batch in enumerate(iterator):
                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        input, names = batch[0], batch[1]
                    else:
                        input = batch
                        names = None
                    input = input.cuda(non_blocking=True)
                    if getattr(config, "eval_crop_size", None) is not None:
                        input = _center_crop_tensor(input, int(config.eval_crop_size))
                    if args.debug_eval_shapes and batch_idx == 0 and _is_main_process():
                        logger.info(f"[{split_name}] input shape: {tuple(input.shape)}")
                    start_time = time.time()

                    if getattr(config, "eval_crop_size", None) is None and bool(args.eval_full_res):
                        if bool(args.eval_direct_full_res):
                            down = int(getattr(config, "downsample", 4))
                            pad_multiple = int(2**down)
                            recon_image, CBR, SNR_out, mse, loss_G = _reconstruct_fullres_direct(
                                net,
                                input,
                                snr=int(SNR),
                                rate=int(rate),
                                amp_enabled=bool(config.amp),
                                pad_multiple=pad_multiple,
                            )
                        else:
                            recon_image, CBR, SNR_out = _reconstruct_fullres_tiled(
                                net,
                                input,
                                snr=int(SNR),
                                rate=int(rate),
                                tile_size=int(args.eval_tile_size),
                                overlap=int(args.eval_tile_overlap),
                                amp_enabled=bool(config.amp),
                            )
                            mse = ((input * 255.0 - recon_image * 255.0) ** 2).mean()
                            net_module = net.module if hasattr(net, "module") else net
                            loss_G = net_module.distortion_loss.forward(input, recon_image).mean()
                    else:
                        with torch.amp.autocast("cuda", enabled=config.amp):
                            recon_image, CBR, SNR_out, mse, loss_G = net(input, SNR, rate)

                    elapsed.update(time.time() - start_time)
                    losses.update(float(loss_G.item()))
                    cbrs.update(CBR)
                    snrs.update(SNR_out)
                    if mse.item() > 0:
                        psnr = 10 * (torch.log(255.0 * 255.0 / mse) / np.log(10))
                        psnrs.update(psnr.item())
                        with torch.amp.autocast("cuda", enabled=False):
                            msssim = 1 - CalcuSSIM(
                                input.float(),
                                recon_image.clamp(0.0, 1.0).float(),
                            ).mean().item()
                        msssims.update(msssim)
                    else:
                        psnrs.update(100)
                        msssims.update(100)

                    if save_images:
                        recon_dir = os.path.join(
                            config.output_test,
                            "recon",
                            config.run_id,
                            f"snr{SNR}_rate{rate}",
                        )
                        os.makedirs(recon_dir, exist_ok=True)

                        # Save images in [0,1] range to avoid color/brightness issues.
                        recon_save = recon_image.detach().clamp(0.0, 1.0)
                        gt_save = input.detach().clamp(0.0, 1.0)

                        # Names are provided by dataset (DIV2K/Kodak). For safety, fall back to index.
                        if isinstance(names, (list, tuple)):
                            names = list(names)
                        elif names is None:
                            names = []
                        else:
                            names = [str(names)]
                        if len(names) != recon_save.size(0):
                            # align length to batch size
                            if len(names) < recon_save.size(0):
                                names = list(names) + [
                                    f"idx{batch_idx}_{i}"
                                    for i in range(len(names), recon_save.size(0))
                                ]
                            else:
                                names = list(names)[: recon_save.size(0)]

                        max_save = min(16, recon_save.size(0))
                        for i_img, name in enumerate(names[:max_save]):
                            base = os.path.splitext(os.path.basename(str(name)))[0] or f"idx{batch_idx}_{i_img}"
                            recon_path = _unique_path(os.path.join(recon_dir, f"reconstructed_image_{base}.png"))
                            gt_path = _unique_path(os.path.join(recon_dir, f"original_image_{base}.png"))
                            cmp_path = _unique_path(os.path.join(recon_dir, f"compare_{base}.png"))

                            torchvision.utils.save_image(recon_save[i_img : i_img + 1], recon_path)
                            torchvision.utils.save_image(gt_save[i_img : i_img + 1], gt_path)
                            torchvision.utils.save_image(
                                torch.cat([gt_save[i_img : i_img + 1], recon_save[i_img : i_img + 1]], dim=3),
                                cmp_path,
                            )

                    if tqdm is not None and _is_main_process() and isinstance(iterator, tqdm):
                        if (batch_idx == 0) or ((batch_idx + 1) % config.print_step == 0):
                            postfix = dict(
                                loss=f"{losses.avg:.3f}",
                                psnr=f"{psnrs.avg:.2f}",
                                msssim=f"{msssims.avg:.3f}",
                            )
                            if bool(getattr(args, "log_cbr", True)):
                                postfix["cbr"] = f"{cbrs.avg:.4f}"
                            iterator.set_postfix(**postfix)

                results_snr[i, j] = snrs.avg
                results_cbr[i, j] = cbrs.avg
                results_psnr[i, j] = psnrs.avg
                results_msssim[i, j] = msssims.avg
                results_loss[i, j] = losses.avg
                parts = [
                    f"[{split_name}] snr={SNR}",
                    f"rate={rate}",
                    f"loss={losses.avg:.4f}",
                    f"psnr={psnrs.avg:.3f}",
                ]
                if bool(getattr(args, "log_cbr", True)):
                    parts.append(f"cbr={cbrs.avg:.6f}")
                parts.append(f"msssim={msssims.avg:.3f}")
                logger.info(" | ".join(parts))
                for m in metrics:
                    m.clear()

    logger.info(f"Finish {split_name}!")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # write curve-friendly csv: one row per (snr, rate)
    if curve_path is None:
        curve_path = os.path.join(config.output_test, "test_curve.csv")
    curve_path = _unique_path(curve_path)
    header = ["snr", "rate", "loss", "psnr", "msssim"]
    if bool(getattr(args, "log_cbr", True)):
        header.append("cbr")
    for i, snr_val in enumerate(multiple_snr):
        for j, rate_val in enumerate(channel_number):
            _append_csv_row(
                curve_path,
                header,
                dict(
                    snr=int(snr_val),
                    rate=int(rate_val),
                    loss=float(results_loss[i, j]),
                    psnr=float(results_psnr[i, j]),
                    msssim=float(results_msssim[i, j]),
                    cbr=float(results_cbr[i, j]),
                ),
            )
def main():
    # Warnings are filtered at import-time; keep this for safety if main.py is imported elsewhere.
    if args.suppress_warnings:
        _bootstrap_warning_filters()

    ddp_info = _setup_distributed()

    seed_torch()
    logger = logger_configuration(config, save_log=False)
    if not _is_main_process():
        # Silence non-rank0 logs.
        logger.setLevel(logging.ERROR)
        for h in list(logger.handlers):
            h.setLevel(logging.ERROR)

    if _is_main_process():
        logger.info(
            "Run config | "
            f"model={args.model} size={args.model_size} channel={args.channel_type} "
            f"snr={args.multiple_snr} C={args.C} "
            f"batch={config.batch_size} accum={config.grad_accum} amp={config.amp} "
            f"epochs={config.tot_epoch} val_freq={getattr(config, 'val_freq', None)} "
            f"dpr={getattr(config, 'drop_path_rate', 0.0):g} "
            f"ls={getattr(config, 'layer_scale', None)} "
            f"ls_conv={getattr(config, 'layer_scale_conv', None)} "
            f"eval_crop={getattr(config, 'eval_crop_size', None)} "
            f"output={config.output_dir} "
            f"ddp={ddp_info.get('distributed', False)} world_size={ddp_info.get('world_size', 1)}"
        )
    torch.manual_seed(seed=config.seed)

    net = MambaVisionJSCC(args, config).to(config.device)

    # Load model weights for test-only runs or finetuning.
    if not args.training and not bool(getattr(args, "profile_model", False)) and not args.ckpt:
        raise ValueError(
            "Test-only run requires `--ckpt /path/to/*_final.pth` (a saved model state_dict). "
            "If you intended to train, add `--training`."
        )
    if args.ckpt:
        if _is_main_process():
            logger.info(f"Loading checkpoint: {args.ckpt} (strict={bool(args.ckpt_strict)})")
        if not os.path.isfile(args.ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {args.ckpt!r}")
        state_dict = _load_checkpoint_state_dict(args.ckpt)
        incompatible = net.load_state_dict(state_dict, strict=bool(args.ckpt_strict))
        if _is_main_process():
            missing = getattr(incompatible, "missing_keys", [])
            unexpected = getattr(incompatible, "unexpected_keys", [])
            logger.info(
                f"Checkpoint loaded. missing_keys={len(missing)} unexpected_keys={len(unexpected)}"
            )
            if (not bool(args.ckpt_strict)) and (missing or unexpected):
                if missing:
                    logger.info(f"Missing keys (first 20): {missing[:20]}")
                if unexpected:
                    logger.info(f"Unexpected keys (first 20): {unexpected[:20]}")
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    # Quick SSM directionality hint (based on the local MambaVisionMixer implementation).
    if _is_main_process():
        try:
            from net.modules import MambaVisionMixer as _MVM  # type: ignore

            # Heuristic: local implementation uses conv1d padding='same' (non-causal) and does not run a reverse scan,
            # so it is NOT a bidirectional selective scan.
            logger.info("SSM check: MambaVisionMixer uses padding='same' (non-causal conv); selective scan is single-pass (not bidirectional).")
        except Exception as e:
            logger.info(f"SSM check skipped: {e!r}")
    if ddp_info.get("distributed", False) and bool(getattr(args, "sync_bn", False)):
        if torch.cuda.is_available():
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
            if _is_main_process():
                logger.info("DDP: converted BatchNorm -> SyncBatchNorm")
        else:
            if _is_main_process():
                logger.info("DDP: --sync-bn ignored (CUDA not available)")
    if ddp_info.get("distributed", False):
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[ddp_info["local_rank"]] if torch.cuda.is_available() else None,
            output_device=ddp_info["local_rank"] if torch.cuda.is_available() else None,
            broadcast_buffers=bool(getattr(args, "sync_bn", False)),
            find_unused_parameters=bool(getattr(args, "ddp_find_unused_parameters", True)),
        )

    if bool(getattr(args, "profile_model", False)) or bool(getattr(args, "profile_only", False)):
        profile_snr = int(str(args.multiple_snr).split(",")[0])
        profile_rate = int(str(args.C).split(",")[0])
        if ddp_info.get("distributed", False):
            if _is_main_process():
                _profile_model_and_save(net, config, logger, profile_snr, profile_rate)
            dist.barrier()
        else:
            _profile_model_and_save(net, config, logger, profile_snr, profile_rate)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if bool(getattr(args, "profile_only", False)):
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
            return

    train_loader, val_loader, test_loader = get_loader(args, config)
    optimizer = _build_optimizer(net, config)
    scheduler = _build_scheduler(optimizer, config)
    global_step = 0
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp)
    if _is_main_process():
        logger.info(
            f"AMP={config.amp}, grad_accum={config.grad_accum}, batch_size={config.batch_size} | "
            f"lr={config.learning_rate:g} wd={getattr(config, 'weight_decay', 0.0):g} "
            f"sched={getattr(config, 'scheduler', 'none')} warmup={getattr(config, 'warmup_epochs', 0)}"
        )

    epoch_metrics_path = _unique_path(
        os.path.join(config.output_train, "epoch_metrics.csv")
    )
    epoch_header = ["epoch", "split", "loss", "psnr", "msssim", "snr"]
    if bool(getattr(args, "log_cbr", True)):
        epoch_header.insert(5, "cbr")

    if args.training:
        for epoch in range(config.tot_epoch):
            # Warmup pretrain: bypass channel for the first N epochs to stabilize reconstruction.
            if int(getattr(config, "pretrain_no_channel_epochs", 0)) > 0:
                config.pass_channel = bool(epoch >= int(config.pretrain_no_channel_epochs))
                if _is_main_process() and epoch == 0:
                    logger.info(
                        f"[warmup] pretrain_no_channel_epochs={int(config.pretrain_no_channel_epochs)} "
                        "(channel bypass ON)"
                    )
                if _is_main_process() and epoch == int(config.pretrain_no_channel_epochs):
                    logger.info("[warmup] channel bypass OFF (training with noise)")

            global_step, train_metrics = train_one_epoch(
                net, optimizer, train_loader, epoch, logger, global_step, scaler, config
            )
            if _is_main_process():
                _append_csv_row(
                    epoch_metrics_path,
                    epoch_header,
                    dict(epoch=epoch + 1, split="train", **train_metrics),
                )
            if getattr(config, "val_freq", None):
                if (epoch + 1) % int(config.val_freq) == 0:
                    # fixed validation setting (deterministic curve)
                    val_snr = int(args.multiple_snr.split(",")[0])
                    val_rate = int(args.C.split(",")[0])
                    if torch.cuda.is_available() and bool(getattr(args, "empty_cache_before_eval", False)):
                        torch.cuda.empty_cache()
                    val_metrics = evaluate_fixed(
                        net, val_loader, logger, val_snr, val_rate, split_name="val"
                    )
                    if _is_main_process() and val_metrics is not None:
                        _append_csv_row(
                            epoch_metrics_path,
                            epoch_header,
                            dict(epoch=epoch + 1, split="val", **val_metrics),
                        )
            if getattr(config, "save_model_freq", None):
                if (epoch + 1) % int(config.save_model_freq) == 0:
                    if _is_main_process():
                        model_to_save = net.module if hasattr(net, "module") else net
                        save_model(
                            model_to_save,
                            save_path=_unique_path(
                                os.path.join(config.models, f"{config.filename}_EP{epoch + 1}.pth")
                            ),
                        )

            if scheduler is not None:
                # Guard against stepping scheduler before any optimizer.step() happened
                # (e.g. AMP overflow skipping all steps in early epochs).
                if getattr(optimizer, "_step_count", 0) > 0:
                    scheduler.step()

        # save final checkpoint after training
        if _is_main_process():
            model_to_save = net.module if hasattr(net, "module") else net
            save_model(
                model_to_save,
                save_path=_unique_path(os.path.join(config.models, f"{config.filename}_final.pth")),
            )

        # training finished: run one final test on Kodak24
        test_curve_path = _unique_path(os.path.join(config.output_test, "test_curve.csv"))
        if torch.cuda.is_available() and bool(getattr(args, "empty_cache_before_test", False)):
            torch.cuda.empty_cache()
        # Call on all ranks: `test()` internally runs only on rank0 and uses barriers
        # to keep DDP ranks in sync and avoid NCCL timeouts.
        test(
            net,
            test_loader,
            logger,
            save_images=True,
            split_name="test",
            curve_path=test_curve_path,
        )
    else:
        test_curve_path = _unique_path(os.path.join(config.output_test, "test_curve.csv"))
        if torch.cuda.is_available() and bool(getattr(args, "empty_cache_before_test", False)):
            torch.cuda.empty_cache()
        test(
            net,
            test_loader,
            logger,
            save_images=True,
            split_name="test",
            curve_path=test_curve_path,
        )

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
