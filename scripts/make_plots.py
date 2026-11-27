"""Make the paper figures from training logs and results/*.csv.

Usage:
    python scripts/make_plots.py [--train-dir DIR] [--results-dir results] [--out-dir paper/figures]

Figures (each as vector PDF and PNG; any figure with no data is skipped with a message):
    loss_curves       training loss and learning rate vs step, one line per CSV in --train-dir
                      (columns step, loss, lr, steps_per_sec)
    success_vs_step   success vs checkpoint_step, one line per format (results with checkpoint_step)
    success_by_format success per format, one marker per split (results with format)
    success_vs_point_error  success vs point_error_cm, one line per format (results with point_error_cm)

Success is shown with Wilson 95% intervals. Needs matplotlib.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_tables as mt  # noqa: E402  (shared loader and Wilson interval)

REPO = Path(__file__).resolve().parent.parent
# Categorical colors in fixed order (validated reference palette, light mode).
COLORS = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
TEXT, GRID = "#191919", "#e5e5e5"


def setup_mpl():
    try:
        import matplotlib
    except ImportError:
        sys.exit("make_plots.py needs matplotlib: pip install matplotlib (it is not in the dem env by default)")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#999999",
        "axes.labelcolor": TEXT, "xtick.color": "#666666", "ytick.color": "#666666",
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.5,
        "legend.frameon": False,
    })
    return plt


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
    print(f"wrote {out_dir / name}.pdf and .png")


def style_of(keys: list[str]) -> dict[str, tuple[str, str]]:
    """Fixed color per entity, by sorted name, so a filter never repaints survivors."""
    return {k: (COLORS[i % len(COLORS)], MARKERS[i % len(MARKERS)]) for i, k in enumerate(sorted(keys))}


def ci_points(rows_by_x: dict[float, list[dict]]):
    xs, ys, lo, hi = [], [], [], []
    for x in sorted(rows_by_x):
        s = mt.success_stats(rows_by_x[x])
        xs.append(x); ys.append(100 * s["rate"])
        lo.append(100 * (s["rate"] - s["lo"])); hi.append(100 * (s["hi"] - s["rate"]))
    return xs, ys, [lo, hi]


# ---------------------------------------------------------------- figures

def load_training(train_dir: Path | None) -> dict[str, dict[str, list[float]]]:
    runs = {}
    if not train_dir or not Path(train_dir).is_dir():
        return runs
    for p in sorted(Path(train_dir).glob("*.csv")):
        cols = defaultdict(list)
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                step, loss = mt.parse_float(r.get("step")), mt.parse_float(r.get("loss"))
                if step is None or loss is None:
                    continue
                cols["step"].append(step); cols["loss"].append(loss)
                cols["lr"].append(mt.parse_float(r.get("lr")))
        if cols["step"]:
            runs[p.stem] = cols
    return runs


def plot_loss(plt, runs, out_dir):
    if not runs:
        print("skip loss_curves: no training CSVs with step and loss (pass --train-dir)")
        return
    if len(runs) > len(COLORS):
        print(f"note: {len(runs)} runs, showing the first {len(COLORS)}")
        runs = dict(list(runs.items())[:len(COLORS)])
    sty = style_of(list(runs))
    has_lr = any(v is not None for c in runs.values() for v in c["lr"])
    fig, axes = plt.subplots(1, 2 if has_lr else 1, figsize=(6.75 if has_lr else 3.25, 2.3), squeeze=False)
    ax = axes[0][0]
    for name, c in runs.items():
        ax.plot(c["step"], c["loss"], color=sty[name][0], label=name)
    ax.set_yscale("log"); ax.set_xlabel("training step"); ax.set_ylabel("loss"); ax.set_title("Training loss")
    if has_lr:
        ax2 = axes[0][1]  # separate panel, never a second y-axis
        for name, c in runs.items():
            pts = [(s, l) for s, l in zip(c["step"], c["lr"]) if l is not None]
            if pts:
                ax2.plot(*zip(*pts), color=sty[name][0], label=name)
        ax2.set_xlabel("training step"); ax2.set_ylabel("learning rate"); ax2.set_title("Learning rate")
        ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    if len(runs) > 1:
        ax.legend()
    fig.tight_layout()
    save(fig, out_dir, "loss_curves"); plt.close(fig)


def plot_success_vs_step(plt, rows, out_dir):
    rows = [r for r in rows if mt.parse_float(r.get("checkpoint_step")) is not None]
    if not rows:
        print("skip success_vs_step: no results with checkpoint_step")
        return
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r.get("format") or "all"][mt.parse_float(r["checkpoint_step"])].append(r)
    sty = style_of(list(by))
    fig, ax = plt.subplots(figsize=(3.25, 2.3))
    for f in sorted(by):
        xs, ys, err = ci_points(by[f])
        ax.errorbar(xs, ys, yerr=err, color=sty[f][0], marker=sty[f][1], markersize=4, capsize=2, label=f)
    ax.set_xlabel("training step"); ax.set_ylabel("success (%)"); ax.set_ylim(0, 100)
    ax.set_title("Success vs training step")
    if len(by) > 1:
        ax.legend()
    save(fig, out_dir, "success_vs_step"); plt.close(fig)


def plot_success_by_format(plt, rows, out_dir):
    rows = [r for r in rows if r.get("format")]
    if not rows:
        print("skip success_by_format: no results with format")
        return
    fmts = sorted({r["format"] for r in rows})
    splits = sorted({r.get("split") or "all" for r in rows})
    sty = style_of(splits)
    fig, ax = plt.subplots(figsize=(3.25, 2.3))
    width = 0.6 / max(1, len(splits))
    for j, s in enumerate(splits):
        for i, f in enumerate(fmts):
            sub = [r for r in rows if r["format"] == f and (r.get("split") or "all") == s]
            if not sub:
                continue
            st = mt.success_stats(sub)
            x = i + (j - (len(splits) - 1) / 2) * width
            ax.errorbar([x], [100 * st["rate"]], yerr=[[100 * (st["rate"] - st["lo"])], [100 * (st["hi"] - st["rate"])]],
                        color=sty[s][0], marker=sty[s][1], markersize=5, capsize=2, linestyle="none",
                        label=s if i == 0 else None)
    ax.set_xticks(range(len(fmts)), fmts, rotation=20, ha="right")
    ax.set_ylabel("success (%)"); ax.set_ylim(0, 100); ax.set_title("Success by command format")
    ax.grid(axis="x", visible=False)
    if len(splits) > 1:
        ax.legend()
    save(fig, out_dir, "success_by_format"); plt.close(fig)


def plot_success_vs_point_error(plt, rows, out_dir):
    rows = [r for r in rows if mt.parse_float(r.get("point_error_cm")) is not None]
    if not rows:
        print("skip success_vs_point_error: no results with point_error_cm")
        return
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r.get("format") or "all"][mt.parse_float(r["point_error_cm"])].append(r)
    sty = style_of(list(by))
    fig, ax = plt.subplots(figsize=(3.25, 2.3))
    for f in sorted(by):
        xs, ys, err = ci_points(by[f])
        ax.errorbar(xs, ys, yerr=err, color=sty[f][0], marker=sty[f][1], markersize=4, capsize=2, label=f)
    ax.set_xlabel("point error (cm)"); ax.set_ylabel("success (%)"); ax.set_ylim(0, 100)
    ax.set_title("Success vs point error")
    if len(by) > 1:
        ax.legend()
    save(fig, out_dir, "success_vs_point_error"); plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train-dir", type=Path, default=None, help="folder of training log CSVs")
    ap.add_argument("--results-dir", type=Path, default=REPO / "results")
    ap.add_argument("--out-dir", type=Path, default=REPO / "paper" / "figures")
    a = ap.parse_args(argv)
    plt = setup_mpl()
    rows, _ = mt.load_rows(a.results_dir) if Path(a.results_dir).is_dir() else ([], [])
    plot_loss(plt, load_training(a.train_dir), a.out_dir)
    plot_success_vs_step(plt, rows, a.out_dir)
    plot_success_by_format(plt, rows, a.out_dir)
    plot_success_vs_point_error(plt, rows, a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
