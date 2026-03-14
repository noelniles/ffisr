"""
evaluate.py — Batch evaluation across link regimes for FFISR whitepaper.

Usage:
    python eval/evaluate.py --dataset /path/to/visdrone/sequences \
                            --model yolov8s.pt \
                            --out eval/results

Produces:
    results/summary.csv          — per-sequence × per-regime raw numbers
    results/summary_table.tex    — LaTeX table ready to \\input{}
    results/fig_bandwidth.pdf    — bandwidth savings by sequence per regime
    results/fig_throughput.pdf   — semantic vs baseline kbps per regime
    results/fig_drops.pdf        — packet drop rate vs link budget
    results/fig_power.pdf        — cumulative transmission proxy per regime
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Link regime definitions matching whitepaper Table 1
# ---------------------------------------------------------------------------
REGIMES: list[dict] = [
    {
        "label": "A — 5000 kbps",
        "link_kbps": 5000,
        "packet_loss": 0.02,
        "rf_blackout_s": 0.0,
        "short": "A",
    },
    {
        "label": "B — 1000 kbps",
        "link_kbps": 1000,
        "packet_loss": 0.02,
        "rf_blackout_s": 5.0,
        "short": "B",
    },
    {
        "label": "C — 200 kbps",
        "link_kbps": 200,
        "packet_loss": 0.02,
        "rf_blackout_s": 10.0,
        "short": "C",
    },
]

SEMANTIC_RATIO = 0.65


def _run_all(
    dataset_dir: str,
    model_path: str,
    tracker: str,
    conf: float,
    max_sequences: int,
    width: int,
    height: int,
) -> list[dict]:
    """Run batch evaluation for every regime; return flat list of row dicts."""
    import importlib
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.batch import _discover_sequences, _run_sequence

    try:
        ultralytics = importlib.import_module("ultralytics")
    except ImportError:
        sys.exit("ultralytics not installed — run: pip install ultralytics")

    _name = Path(model_path).stem.lower()
    if "rtdetr" in _name or "rt-detr" in _name:
        model = ultralytics.RTDETR(model_path)
    else:
        model = ultralytics.YOLO(model_path)

    sequences = _discover_sequences(dataset_dir)[:max_sequences]
    if not sequences:
        sys.exit(f"No image sequences found under: {dataset_dir}")

    print(f"Found {len(sequences)} sequence(s). Running {len(REGIMES)} regime(s)…")

    rows: list[dict] = []
    for regime in REGIMES:
        print(f"\n  Regime {regime['short']} ({regime['label']})")
        for seq in sequences:
            print(f"    {seq.name} … ", end="", flush=True)
            r = _run_sequence(
                seq_dir=seq,
                model=model,
                tracker=tracker,
                conf=conf,
                link_kbps=regime["link_kbps"],
                packet_loss=regime["packet_loss"],
                semantic_ratio=SEMANTIC_RATIO,
                width=width,
                height=height,
            )
            if r.get("error"):
                print(f"SKIP ({r['error']})")
                continue
            rows.append(
                {
                    "regime": regime["short"],
                    "regime_label": regime["label"],
                    "link_kbps": regime["link_kbps"],
                    "sequence": r["sequence"],
                    "frames": r["frames"],
                    "unique_tracks": r["unique_tracks"],
                    "semantic_kbps": r["semantic_kbps"],
                    "baseline_kbps": r["baseline_kbps"],
                    "sem_attempted_kbps": r["sem_attempted_kbps"],
                    "base_attempted_kbps": r["base_attempted_kbps"],
                    "bandwidth_savings_pct": r["bandwidth_savings_pct"],
                    "semantic_drop_pct": r["semantic_drop_pct"],
                    "baseline_drop_pct": r["baseline_drop_pct"],
                    "power_proxy_semantic_mbits": r["power_proxy_semantic_mbits"],
                    "power_proxy_baseline_mbits": r["power_proxy_baseline_mbits"],
                }
            )
            print(
                f"done  bw_savings={r['bandwidth_savings_pct']:.1f}%  "
                f"sem={r['sem_attempted_kbps']:.1f} kbps  "
                f"base={r['base_attempted_kbps']:.1f} kbps"
            )

    return rows


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV  → {path}")


def _regime_summary(rows: list[dict]) -> dict[str, dict]:
    """Aggregate per-regime averages from flat rows."""
    from collections import defaultdict

    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[r["regime"]].append(r)

    out: dict[str, dict] = {}
    for regime in REGIMES:
        key = regime["short"]
        bucket = buckets.get(key, [])
        if not bucket:
            continue
        n = len(bucket)
        out[key] = {
            "label": regime["label"],
            "link_kbps": regime["link_kbps"],
            "n": n,
            # attempted (link-independent) — used for savings metric
            "avg_sem_attempted_kbps": sum(r["sem_attempted_kbps"] for r in bucket) / n,
            "avg_base_attempted_kbps": sum(r["base_attempted_kbps"] for r in bucket) / n,
            # delivered (link-constrained) — used for throughput / drop charts
            "avg_semantic_kbps": sum(r["semantic_kbps"] for r in bucket) / n,
            "avg_baseline_kbps": sum(r["baseline_kbps"] for r in bucket) / n,
            "avg_bandwidth_savings_pct": sum(r["bandwidth_savings_pct"] for r in bucket) / n,
            "avg_semantic_drop_pct": sum(r["semantic_drop_pct"] for r in bucket) / n,
            "avg_baseline_drop_pct": sum(r["baseline_drop_pct"] for r in bucket) / n,
            "total_sem_mbits": sum(r["power_proxy_semantic_mbits"] for r in bucket),
            "total_base_mbits": sum(r["power_proxy_baseline_mbits"] for r in bucket),
        }
        total_base = out[key]["total_base_mbits"]
        total_sem = out[key]["total_sem_mbits"]
        out[key]["power_savings_pct"] = (
            (1.0 - total_sem / total_base) * 100.0 if total_base > 0 else 0.0
        )
    return out


def _write_latex_table(summary: dict[str, dict], path: Path) -> None:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Regime & Link & Sem.\ del.\ & Base.\ del.\ & BW Savings & "
        r"Sem.\ drop & Base.\ drop \\",
        r"& (kbps) & (kbps, avg) & (kbps, avg) & (\%) & (\%) & (\%) \\",
        r"\midrule",
    ]
    for key in ["A", "B", "C"]:
        if key not in summary:
            continue
        s = summary[key]
        lines.append(
            f"{key} & {s['link_kbps']} & "
            f"{s['avg_semantic_kbps']:.1f} & "
            f"{s['avg_baseline_kbps']:.1f} & "
            f"{s['avg_bandwidth_savings_pct']:.1f} & "
            f"{s['avg_semantic_drop_pct']:.1f} & "
            f"{s['avg_baseline_drop_pct']:.1f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Evaluation results averaged across 10 VisDrone sequences "
        r"for each link regime. BW Savings computed against attempted baseline "
        r"traffic ($1 - B_s^{\text{att}} / B_b^{\text{att}}$). "
        r"Delivered kbps reflects actual link throughput under the token-bucket "
        r"emulator. Semantic drop and baseline drop are packet-level rates.}",
        r"\label{tab:results}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"LaTeX → {path}")


def _make_charts(rows: list[dict], summary: dict[str, dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skipping charts (pip install matplotlib)")
        return

    regime_keys = [r["short"] for r in REGIMES if r["short"] in summary]
    palette = {"A": "#2196F3", "B": "#FF9800", "C": "#F44336"}

    sequences = sorted({r["sequence"] for r in rows})
    x = np.arange(len(sequences))
    bar_w = 0.25

    # ------------------------------------------------------------------
    # Fig 1: Bandwidth savings (%) per sequence, grouped by regime
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(7, len(sequences) * 1.2), 4.5))
    for i, key in enumerate(regime_keys):
        vals = []
        for seq in sequences:
            match = [r["bandwidth_savings_pct"] for r in rows if r["regime"] == key and r["sequence"] == seq]
            vals.append(match[0] if match else 0.0)
        ax.bar(x + i * bar_w, vals, bar_w, label=summary[key]["label"], color=palette[key], alpha=0.85)

    ax.set_xlabel("Sequence")
    ax.set_ylabel("Bandwidth Savings (%)")
    ax.set_title("FFISR Bandwidth Savings by Sequence and Link Regime")
    ax.set_xticks(x + bar_w)
    ax.set_xticklabels(sequences, rotation=30, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    p = out_dir / "fig_bandwidth.pdf"
    fig.savefig(p)
    plt.close(fig)
    print(f"Fig  → {p}")

    # ------------------------------------------------------------------
    # Fig 2: Delivered throughput: semantic vs baseline, per regime
    # Shows how baseline collapses under link constraints while
    # semantic continues to be delivered.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    regime_labels = [summary[k]["label"] for k in regime_keys]
    sem_delivered = [summary[k]["avg_semantic_kbps"] for k in regime_keys]
    base_delivered = [summary[k]["avg_baseline_kbps"] for k in regime_keys]
    sem_attempted = [summary[k]["avg_sem_attempted_kbps"] for k in regime_keys]
    base_attempted = [summary[k]["avg_base_attempted_kbps"] for k in regime_keys]
    xi = np.arange(len(regime_keys))
    bw = 0.2
    ax.bar(xi - 1.5 * bw, base_attempted, bw, label="Baseline needed",
           color="#B0BEC5", alpha=0.7, hatch="//")
    ax.bar(xi - 0.5 * bw, base_delivered, bw, label="Baseline delivered",
           color="#78909C", alpha=0.9)
    ax.bar(xi + 0.5 * bw, sem_attempted, bw, label="Semantic needed",
           color="#90CAF9", alpha=0.7, hatch="//")
    ax.bar(xi + 1.5 * bw, sem_delivered, bw, label="Semantic delivered",
           color="#1E88E5", alpha=0.9)
    ax.set_xticks(xi)
    ax.set_xticklabels(regime_labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Average Throughput (kbps)")
    ax.set_title("Needed vs Delivered Throughput by Regime")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p = out_dir / "fig_throughput.pdf"
    fig.savefig(p)
    plt.close(fig)
    print(f"Fig  → {p}")

    # ------------------------------------------------------------------
    # Fig 3: Packet drop rate vs link budget
    # The key resilience story: baseline drop spikes at low budget,
    # semantic drop stays flat.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    link_budgets = [summary[k]["link_kbps"] for k in regime_keys]
    sem_drops = [summary[k]["avg_semantic_drop_pct"] for k in regime_keys]
    base_drops = [summary[k]["avg_baseline_drop_pct"] for k in regime_keys]
    ax.plot(link_budgets, base_drops, "o--", color="#78909C",
            linewidth=2, label="Baseline drop")
    ax.plot(link_budgets, sem_drops, "o-", color="#1E88E5",
            linewidth=2, label="Semantic (FFISR) drop")
    for k_idx, key in enumerate(regime_keys):
        ax.annotate(
            f" {key}",
            (link_budgets[k_idx], base_drops[k_idx]),
            fontsize=8, color="#78909C",
        )
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Link Budget (kbps) — degrading left")
    ax.set_ylabel("Avg Packet Drop Rate (%)")
    ax.set_title("Resilience: Drop Rate as Link Degrades")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.legend(fontsize=8)
    ax.set_ylim(-2, 105)
    fig.tight_layout()
    p = out_dir / "fig_drops.pdf"
    fig.savefig(p)
    plt.close(fig)
    print(f"Fig  → {p}")

    # ------------------------------------------------------------------
    # Fig 4: Attempted transmission volume — semantic vs baseline
    # Illustrates the raw data reduction before any link constraints.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    total_sem = [summary[k]["total_sem_mbits"] for k in regime_keys]
    total_base = [summary[k]["total_base_mbits"] for k in regime_keys]
    xi = np.arange(len(regime_keys))
    bars_base = ax.bar(xi - bw * 2, total_base, bw * 1.5,
                       label="Baseline (full video)",
                       color="#78909C", alpha=0.9)
    bars_sem = ax.bar(xi, total_sem, bw * 1.5,
                      label="Semantic (FFISR)",
                      color="#1E88E5", alpha=0.9)
    for bar in bars_base:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.0f}",
                ha="center", va="bottom", fontsize=7)
    for bar in bars_sem:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.0f}",
                ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xi - bw)
    ax.set_xticklabels(regime_labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Total Attempted Transmission (Mbits)")
    ax.set_title("Data Volume: FFISR vs Full-Video Baseline")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "fig_power.pdf"
    fig.savefig(p)
    plt.close(fig)
    print(f"Fig  → {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FFISR batch evaluator for whitepaper figures")
    parser.add_argument("--dataset", required=True, help="Root directory of image sequences")
    parser.add_argument("--model", default="yolov8s.pt", help="Model checkpoint path")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--max-sequences", type=int, default=10)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--out", default="eval/results", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _run_all(
        dataset_dir=args.dataset,
        model_path=args.model,
        tracker=args.tracker,
        conf=args.conf,
        max_sequences=args.max_sequences,
        width=args.width,
        height=args.height,
    )

    if not rows:
        sys.exit("No results produced — check dataset path and model.")

    _write_csv(rows, out_dir / "summary.csv")

    summary = _regime_summary(rows)

    print("\n--- Per-Regime Summary ---")
    for key, s in summary.items():
        print(
            f"  Regime {key} ({s['label']}): "
            f"avg BW savings={s['avg_bandwidth_savings_pct']:.1f}%  "
            f"TX proxy savings={s['power_savings_pct']:.1f}%  "
            f"sem drop={s['avg_semantic_drop_pct']:.1f}%"
        )

    _write_latex_table(summary, out_dir / "summary_table.tex")
    _make_charts(rows, summary, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
