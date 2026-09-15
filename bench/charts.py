"""Generate the README charts from benchmark results.

    python -m bench.charts

Writes PNGs to ``docs/``. Deliberately dependency-light on the plotting side:
matplotlib only, no seaborn, so the figures are reproducible anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

RESULTS = Path(__file__).parent / "results"
DOCS = Path(__file__).parent.parent / "docs"

INK = "#1c1c1e"
MUTED = "#8a8a8e"
GRID = "#e6e6ea"
ACCENT = "#0b6e4f"      # arara
ACCENT_SOFT = "#8fbfae"
WARN = "#c1440e"        # the thing we beat
NEUTRAL = "#c9c9ce"


def _style(ax) -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=10)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _load(name: str):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


# ---------------------------------------------------------------------------
# 1. the chunking / truncation ablation
# ---------------------------------------------------------------------------
ABLATION_ORDER = [
    ("brtaxqa_capped_document_dense", "Capped at 32k, one vector per doc", WARN),
    ("brtaxqa_full_document_dense", "Full documents, one vector per doc", WARN),
    ("brtaxqa_capped_window_dense", "Capped at 32k, fixed windows", NEUTRAL),
    ("brtaxqa_capped_tinyzchunk_dense", "Capped at 32k, tinyzchunk", ACCENT_SOFT),
    ("brtaxqa_capped_tinyzchunk_hybrid", "Capped at 32k, tinyzchunk + BM25", ACCENT_SOFT),
    ("brtaxqa_full_window_dense", "Full documents, fixed windows", NEUTRAL),
    ("brtaxqa_full_paragraph_dense", "Full documents, paragraph splits", NEUTRAL),
    ("brtaxqa_full_tinyzchunk_dense", "Full documents, tinyzchunk", ACCENT_SOFT),
    ("brtaxqa_full_tinyzchunk_hybrid", "Full documents, tinyzchunk + BM25", ACCENT),
    ("brtaxqa_full_tinyzchunk_hybrid_cxm25", "Full documents, + CXM25 rerank", ACCENT),
]


def chart_ablation(records: dict) -> None:
    rows = [(label, records[key]["ndcg_at_10"], color)
            for key, label, color in ABLATION_ORDER if key in records]
    if not rows:
        return
    labels = [r[0] for r in rows][::-1]
    values = [r[1] for r in rows][::-1]
    colors = [r[2] for r in rows][::-1]

    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=200)
    _style(ax)
    bars = ax.barh(labels, values, color=colors, height=0.68, zorder=3)
    for bar, v in zip(bars, values):
        ax.text(v + 0.008, bar.get_y() + bar.get_height() / 2, f"{v:.3f}",
                va="center", ha="left", fontsize=10, color=INK, fontweight="normal")

    best = max(values)
    worst = min(values)
    ax.set_xlim(0, best * 1.18)
    ax.set_xlabel("nDCG@10", color=MUTED, fontsize=10)
    ax.set_title(
        "Chunking a legal corpus beats truncating it by 3.4×",
        fontsize=15, color=INK, fontweight="bold", loc="left", pad=26,
    )
    ax.text(0, 1.045, "BR-TaxQA-R, MTEB-BR · every row changes one thing · 478 statutes, 715 questions",
            transform=ax.transAxes, fontsize=10, color=MUTED)
    ax.legend(handles=[
        Patch(facecolor=WARN, label="one vector per document (the leaderboard's setting)"),
        Patch(facecolor=NEUTRAL, label="formatting heuristics"),
        Patch(facecolor=ACCENT_SOFT, label="tinyzchunk / hybrid"),
        Patch(facecolor=ACCENT, label="full stack"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False,
        fontsize=9, labelcolor=MUTED)

    fig.tight_layout()
    fig.savefig(DOCS / "ablation.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. speed and memory
# ---------------------------------------------------------------------------
def chart_profile(records: list[dict], reopened: list[dict] | None = None) -> None:
    if not records:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), dpi=200)

    for ax, key, title, unit in (
        (axes[0], "build_s", "Index build", "seconds"),
        (axes[1], "latency_p50_ms", "Query latency", "ms (p50)"),
    ):
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        for persistent, color, label in ((False, ACCENT, "in memory"), (True, ACCENT_SOFT, "out of core")):
            pts = sorted((r["documents"], r[key]) for r in records if r["persistent"] == persistent)
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", markersize=5, linewidth=2.0, color=color, label=label, zorder=3)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=12, color=INK, fontweight="normal", loc="left", pad=8)
        ax.set_xlabel("documents", color=MUTED, fontsize=9)
        ax.set_ylabel(unit, color=MUTED, fontsize=9)

    ax = axes[2]
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    if reopened:
        pts = sorted((r["documents"], r["rss_to_serve_mb"]) for r in reopened)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", markersize=5, linewidth=2.0, color=ACCENT, zorder=3)
        base = ys[0]
        ax.annotate(f"{base:.0f} MB fixed\n(python + numpy + model)",
                    xy=(xs[0], base), xytext=(xs[0] * 1.3, base * 0.86),
                    fontsize=8.5, color=MUTED,
                    arrowprops=dict(arrowstyle="-", color=GRID, linewidth=1))
        ax.set_ylim(0, max(ys) * 1.25)
    ax.set_xscale("log")
    ax.set_title("Memory to serve a reopened index", fontsize=12, color=INK,
                 fontweight="normal", loc="left", pad=8)
    ax.set_xlabel("documents", color=MUTED, fontsize=9)
    ax.set_ylabel("peak RSS (MB)", color=MUTED, fontsize=9)

    fig.suptitle(
        "One core, low-millisecond queries, and a per-document cost under two kilobytes",
        fontsize=14, color=INK, fontweight="bold", x=0.007, ha="left", y=1.06,
    )
    fig.tight_layout()
    fig.savefig(DOCS / "scaling.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. where arara sits on the public leaderboard
# ---------------------------------------------------------------------------
def chart_leaderboard() -> None:
    path = RESULTS / "leaderboard_comparison.json"
    if not path.exists():
        return
    blob = json.loads(path.read_text())
    lb, ours = blob.get("leaderboard", {}), blob.get("arara", {})
    tasks = [("FaQuADIR", "FaQuADIR"), ("BRTaxQAR", "BRTaxQAR"), ("JurisTCU", "JurisTCU"),
             ("Quati", "Quati"), ("FaqBacenRetrieval", "FaqBacen")]
    rows = []
    for task, label in tasks:
        models = lb.get(task, {})
        mine = ours.get(task, {})
        if not models or not mine:
            continue
        best = max(mine.values())
        rows.append((label, sorted(models.values()), best))
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=200)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    for i, (label, scores, mine) in enumerate(rows):
        y = len(rows) - i - 1
        ax.scatter(scores, [y] * len(scores), s=18, color=NEUTRAL, alpha=0.85,
                   zorder=2, label="MTEB-BR leaderboard models" if i == 0 else None)
        ax.scatter([mine], [y], s=110, color=ACCENT, zorder=4, marker="D",
                   label="arara-rag (best mode)" if i == 0 else None)
        beaten = sum(1 for s in scores if s < mine)
        ax.text(mine, y + 0.26, f"{beaten}/{len(scores)}", ha="center", fontsize=9,
                color=ACCENT, fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], color=INK, fontsize=10)
    ax.set_xlabel("main metric (nDCG@10)", color=MUTED, fontsize=10)
    ax.set_title("A 200 MB CPU stack against 96 published embedding models",
                 fontsize=15, color=INK, fontweight="bold", loc="left", pad=26)
    ax.text(0, 1.05, "each grey dot is a model on the MTEB-BR leaderboard; labels count how many arara beats",
            transform=ax.transAxes, fontsize=10, color=MUTED)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="lower right")
    fig.tight_layout()
    fig.savefig(DOCS / "leaderboard.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    ablation = _load("summary_brtaxqa.json")
    if ablation:
        chart_ablation({r["experiment"]: r for r in ablation})
        print("wrote docs/ablation.png")
    profile = _load("profile.json")
    if profile:
        chart_profile(profile, _load("profile_reopen.json"))
        print("wrote docs/scaling.png")
    chart_leaderboard()
    if (RESULTS / "leaderboard_comparison.json").exists():
        print("wrote docs/leaderboard.png")


if __name__ == "__main__":
    main()
