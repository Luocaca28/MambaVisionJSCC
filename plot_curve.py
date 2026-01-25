import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def _try_set_style():
    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        try:
            plt.style.use(style)
            return
        except Exception:
            pass


def _ema(values: list[float], alpha: float) -> list[float]:
    if not values:
        return []
    a = max(0.0, min(1.0, float(alpha)))
    out = [float(values[0])]
    for v in values[1:]:
        out.append(a * float(v) + (1.0 - a) * out[-1])
    return out


def _moving_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    w = max(1, int(window))
    out: list[float] = []
    s = 0.0
    q: list[float] = []
    for v in values:
        fv = float(v)
        q.append(fv)
        s += fv
        if len(q) > w:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def _smooth(values: list[float], method: str, window: int, alpha: float) -> list[float]:
    method = (method or "ema").lower()
    if method in ("none", "raw", "off"):
        return values
    if method in ("ma", "moving", "moving_average"):
        return _moving_average(values, window=window)
    return _ema(values, alpha=alpha)


def _linear_interpolate_int_x(xs: list[int], ys: list[float]) -> tuple[list[int], list[float]]:
    """
    Given sparse integer x points, linearly interpolate y on every integer step.
    Keeps endpoints unchanged; does not extrapolate beyond the min/max x.
    """
    if not xs or len(xs) != len(ys) or len(xs) < 2:
        return xs, ys
    pts = sorted(zip(xs, ys), key=lambda t: t[0])
    xs_sorted = [int(p[0]) for p in pts]
    ys_sorted = [float(p[1]) for p in pts]
    x_min, x_max = xs_sorted[0], xs_sorted[-1]
    dense_xs = list(range(x_min, x_max + 1))

    dense_ys: list[float] = []
    seg = 0
    for x in dense_xs:
        while seg < len(xs_sorted) - 2 and x > xs_sorted[seg + 1]:
            seg += 1
        x0, x1 = xs_sorted[seg], xs_sorted[seg + 1]
        y0, y1 = ys_sorted[seg], ys_sorted[seg + 1]
        if x1 == x0:
            dense_ys.append(y1)
        else:
            t = (x - x0) / (x1 - x0)
            dense_ys.append(y0 * (1.0 - t) + y1 * t)
    return dense_xs, dense_ys


def _read_csv_rows(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while True:
        candidate = f"{base}_{i}{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1


def _latest_csv(path_dir: str, prefix: str):
    if not os.path.isdir(path_dir):
        return None
    candidates = []
    for name in os.listdir(path_dir):
        if name.startswith(prefix) and name.endswith(".csv"):
            full = os.path.join(path_dir, name)
            try:
                candidates.append((os.path.getmtime(full), full))
            except OSError:
                pass
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def plot_curve(
    output_dir: str = "./output",
    train_csv: str | None = None,
    smooth: str = "none",
    smooth_window: int = 15,
    smooth_alpha: float = 0.12,
    val_interp: str = "linear",
):
    output_dir = os.path.abspath(output_dir)
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    if train_csv is None:
        train_csv = _latest_csv(train_dir, "epoch_metrics")
    else:
        train_csv = os.path.abspath(train_csv)
    test_csv = _latest_csv(test_dir, "test_curve")
    picture_dir = os.path.join(output_dir, "picture")
    os.makedirs(picture_dir, exist_ok=True)
    _try_set_style()

    # 1) train/val loss curve (loss vs epoch)
    if train_csv is None:
        raise FileNotFoundError(
            f"No epoch metrics csv found under: {train_dir} (expected epoch_metrics*.csv)"
        )
    if not os.path.exists(train_csv):
        raise FileNotFoundError(train_csv)
    rows = _read_csv_rows(train_csv)
    by_split = defaultdict(list)
    by_split_psnr = defaultdict(list)
    for r in rows:
        try:
            epoch = int(float(r["epoch"]))
        except Exception:
            continue
        split = r.get("split", "train")
        try:
            loss = float(r["loss"])
            by_split[split].append((epoch, loss))
        except Exception:
            pass
        try:
            psnr = float(r["psnr"])
            by_split_psnr[split].append((epoch, psnr))
        except Exception:
            pass

    plt.figure(figsize=(9, 5))
    smooth_mode = (smooth or "none").lower()
    val_interp_mode = (val_interp or "none").lower()
    for split in ("train", "val"):
        if split not in by_split:
            continue
        pts = sorted(by_split[split], key=lambda x: x[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        # Raw-only is the most faithful view; keep it clean (no big markers).
        if smooth_mode in ("none", "raw", "off"):
            # Keep both curves as lines (no markers). If val is not interpolated,
            # use a dashed line to visually indicate sparse evaluation.
            if split == "val" and val_interp_mode in ("none", "off", "raw"):
                ls = "--"
            else:
                ls = "-"
            if split == "val" and val_interp_mode in ("linear", "interp", "interpolate"):
                xs_plot, ys_plot = _linear_interpolate_int_x(xs, ys)
            else:
                xs_plot, ys_plot = xs, ys
            plt.plot(
                xs_plot,
                ys_plot,
                linewidth=2.0,
                linestyle=ls,
                alpha=0.95,
                marker=None,
                solid_capstyle="round",
                label=split,
            )
        else:
            ys_s = _smooth(ys, method=smooth, window=smooth_window, alpha=smooth_alpha)
            plt.plot(xs, ys, linewidth=1.0, alpha=0.18, label=f"{split} (raw)")
            plt.plot(xs, ys_s, linewidth=2.6, alpha=0.95, label=f"{split} (smooth)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train/Val Loss Curve")
    plt.grid(True, alpha=0.25)
    plt.legend()
    loss_path = _unique_path(os.path.join(picture_dir, "loss_curve.png"))
    plt.tight_layout()
    plt.savefig(loss_path, dpi=200)
    plt.close()

    # 2) train/val PSNR curve (PSNR vs epoch)
    psnr_curve_path = None
    if by_split_psnr:
        plt.figure(figsize=(9, 5))
        for split in ("train", "val"):
            if split not in by_split_psnr:
                continue
            pts = sorted(by_split_psnr[split], key=lambda x: x[0])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]

            if smooth_mode in ("none", "raw", "off"):
                if split == "val" and val_interp_mode in ("none", "off", "raw"):
                    ls = "--"
                else:
                    ls = "-"
                if split == "val" and val_interp_mode in ("linear", "interp", "interpolate"):
                    xs_plot, ys_plot = _linear_interpolate_int_x(xs, ys)
                else:
                    xs_plot, ys_plot = xs, ys
                plt.plot(
                    xs_plot,
                    ys_plot,
                    linewidth=2.0,
                    linestyle=ls,
                    alpha=0.95,
                    marker=None,
                    solid_capstyle="round",
                    label=split,
                )
            else:
                ys_s = _smooth(ys, method=smooth, window=smooth_window, alpha=smooth_alpha)
                plt.plot(xs, ys, linewidth=1.0, alpha=0.18, label=f"{split} (raw)")
                plt.plot(xs, ys_s, linewidth=2.6, alpha=0.95, label=f"{split} (smooth)")
        plt.xlabel("Epoch")
        plt.ylabel("PSNR (dB)")
        plt.title("Train/Val PSNR Curve")
        plt.grid(True, alpha=0.25)
        plt.legend()
        psnr_curve_path = _unique_path(os.path.join(picture_dir, "psnr_curve.png"))
        plt.tight_layout()
        plt.savefig(psnr_curve_path, dpi=200)
        plt.close()

    # 3) test curve (PSNR vs SNR), one curve per rate
    psnr_snr_path = None
    if test_csv is not None and os.path.exists(test_csv):
        rows = _read_csv_rows(test_csv)
        by_rate = defaultdict(list)
        for r in rows:
            try:
                snr = int(float(r["snr"]))
                rate = int(float(r["rate"]))
                psnr = float(r["psnr"])
            except Exception:
                continue
            by_rate[rate].append((snr, psnr))

        if by_rate:
            plt.figure(figsize=(9, 5))
            for rate in sorted(by_rate.keys()):
                pts = sorted(by_rate[rate], key=lambda x: x[0])
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                plt.plot(xs, ys, marker="o", markersize=4, linewidth=2.2, label=f"rate={rate}")
            plt.xlabel("SNR (dB)")
            plt.ylabel("PSNR (dB)")
            plt.title("Test PSNR vs SNR")
            plt.grid(True, alpha=0.25)
            plt.legend()
            psnr_snr_path = _unique_path(os.path.join(picture_dir, "psnr_snr_curve.png"))
            plt.tight_layout()
            plt.savefig(psnr_snr_path, dpi=200)
            plt.close()

    return {
        "loss_curve": loss_path,
        "psnr_curve": psnr_curve_path,
        "psnr_snr_curve": psnr_snr_path,
    }


def _parse_args():
    p = argparse.ArgumentParser(description="Plot MambaVision_JSCC curves")
    p.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="output directory that contains train/, test/, picture/",
    )
    p.add_argument(
        "--train-csv",
        type=str,
        default=None,
        help="path to a specific epoch_metrics*.csv (otherwise picks the latest under output/train/)",
    )
    p.add_argument(
        "--smooth",
        type=str,
        default="none",
        choices=["ema", "ma", "none"],
        help="loss smoothing method: none (default, raw only), ema, ma (moving average)",
    )
    p.add_argument(
        "--smooth-window",
        type=int,
        default=15,
        help="moving average window (used when --smooth ma)",
    )
    p.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.12,
        help="EMA alpha (used when --smooth ema). Smaller = smoother.",
    )
    p.add_argument(
        "--val-interp",
        type=str,
        default="linear",
        choices=["linear", "none"],
        help="When --smooth none, densify val points by interpolation to draw a smoother val curve.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out = plot_curve(
        args.output_dir,
        train_csv=args.train_csv,
        smooth=args.smooth,
        smooth_window=args.smooth_window,
        smooth_alpha=args.smooth_alpha,
        val_interp=args.val_interp,
    )
    print("Saved:", out["loss_curve"])
    if out.get("psnr_curve"):
        print("Saved:", out["psnr_curve"])
    if out.get("psnr_snr_curve"):
        print("Saved:", out["psnr_snr_curve"])
