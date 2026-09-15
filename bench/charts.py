"""Generate the README charts from benchmark results.

    python -m bench.charts

Writes PNGs to ``docs/``. Deliberately dependency-light on the plotting side:
matplotlib only, no seaborn, so the figures are reproducible anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

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


def chart_ablation(records: dict, nano: dict | None = None) -> None:
    """Grouped bars: the same ablation under each dense backend."""
    rows = [(label, key) for key, label, _ in ABLATION_ORDER if key in records]
    if not rows:
        return
    nano = nano or {}
    nano_key = {key: key.replace("brtaxqa_", "brtaxqa_") for key, _, _ in ABLATION_ORDER}

    labels = [r[0] for r in rows][::-1]
    static_v = [records[r[1]]["ndcg_at_10"] for r in rows][::-1]
    nano_v = [nano.get(r[1], {}).get("ndcg_at_10") for r in rows][::-1]

    fig, ax = plt.subplots(figsize=(10.2, 6.0), dpi=200)
    _style(ax)
    y = np.arange(len(labels))
    h = 0.38
    ax.barh(y + h / 2, static_v, height=h, color=ACCENT_SOFT, zorder=3,
            label="static (default)")
    if any(v is not None for v in nano_v):
        vals = [v if v is not None else 0.0 for v in nano_v]
        ax.barh(y - h / 2, vals, height=h, color=ACCENT, zorder=3, label="nanoE5.c")
        for yi, v in zip(y - h / 2, nano_v):
            if v is not None:
                ax.text(v + 0.008, yi, f"{v:.3f}", va="center", fontsize=9,
                        color=ACCENT, fontweight="bold")
    for yi, v in zip(y + h / 2, static_v):
        ax.text(v + 0.008, yi, f"{v:.3f}", va="center", fontsize=9, color=MUTED)

    best = max([v for v in static_v + (nano_v if nano_v else []) if v is not None])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK, fontsize=10)
    ax.set_xlim(0, best * 1.16)
    ax.set_xlabel("nDCG@10", color=MUTED, fontsize=10)
    ax.set_title("Chunking a legal corpus beats truncating it by 3.4x",
                 fontsize=15, color=INK, fontweight="bold", loc="left", pad=30)
    ax.text(0, 1.045,
            "BR-TaxQA-R, MTEB-BR · every row changes one thing · 478 statutes, 715 questions",
            transform=ax.transAxes, fontsize=10, color=MUTED)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=MUTED, loc="lower right")
    fig.tight_layout()
    fig.savefig(DOCS / "ablation.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def chart_encoder_gain(static: dict, nano: dict) -> None:
    """The dense encoder is what was holding the hybrid back."""
    tasks = [("brtaxqa_capped", "BRTaxQAR"), ("faquadir", "FaQuADIR"),
             ("faq_bacen", "FaqBacen"), ("juristcu", "JurisTCU"), ("quati", "Quati")]
    labels, sd, nd, sh, nh, lex = [], [], [], [], [], []
    for key, name in tasks:
        s_h = static.get(f"{key}_hybrid")
        n_h = nano.get(f"{key}_hybrid")
        if not (s_h and n_h):
            continue
        labels.append(name)
        sd.append(static[f"{key}_dense"]["ndcg_at_10"])
        nd.append(nano[f"{key}_dense"]["ndcg_at_10"])
        sh.append(s_h["ndcg_at_10"])
        nh.append(n_h["ndcg_at_10"])
        lex.append(static[f"{key}_lexical"]["ndcg_at_10"])
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(10.2, 4.8), dpi=200)
    _style(ax)
    x = np.arange(len(labels))
    w = 0.19
    ax.bar(x - 1.5 * w, sd, w, color=NEUTRAL, zorder=3, label="static dense")
    ax.bar(x - 0.5 * w, nd, w, color=ACCENT, zorder=3, label="nanoE5 dense")
    ax.bar(x + 0.5 * w, sh, w, color="#d8d8dd", zorder=3, label="static hybrid")
    ax.bar(x + 1.5 * w, nh, w, color=ACCENT_SOFT, zorder=3, label="nanoE5 hybrid")
    ax.plot(x, lex, linestyle="none", marker="_", markersize=22, markeredgewidth=2.2,
            color=WARN, zorder=5, label="lexical (BM25) ceiling")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=INK, fontsize=10)
    ax.set_ylabel("nDCG@10", color=MUTED, fontsize=10)
    ax.set_ylim(0, max(max(sh), max(nh), max(lex)) * 1.12)
    ax.set_title("A real dense encoder is what makes hybrid retrieval work",
                 fontsize=15, color=INK, fontweight="bold", loc="left", pad=26)
    ax.text(0, 1.05, "bars: dense and hybrid per backend · red dashes: BM25 alone",
            transform=ax.transAxes, fontsize=10, color=MUTED)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=MUTED, ncol=5,
              loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(DOCS / "encoders.png", facecolor="white", bbox_inches="tight")
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
        base, top = ys[0], ys[-1]
        ax.annotate(
            f"{base:.0f} to {top:.0f} MB\n{(top / base - 1):.1%} more for"
            "\n50x the documents",
            xy=(xs[-1], top), xytext=(xs[0] * 1.25, top * 0.72),
            fontsize=8.5, color=MUTED,
            arrowprops=dict(arrowstyle="-", color=GRID, linewidth=1),
        )
        ax.set_ylim(0, max(ys) * 1.25)
    ax.set_xscale("log")
    ax.set_title("Memory to serve a reopened index", fontsize=12, color=INK,
                 fontweight="normal", loc="left", pad=8)
    ax.set_xlabel("documents", color=MUTED, fontsize=9)
    ax.set_ylabel("peak RSS (MB)", color=MUTED, fontsize=9)

    fig.suptitle(
        "One core, exact queries, and a serving footprint that mostly ignores corpus size",
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
    lb = blob.get("leaderboard", {})
    static_path = RESULTS / "static" / "leaderboard_comparison.json"
    if static_path.exists():
        ours = json.loads(static_path.read_text()).get("arara", {})
    else:
        ours = blob.get("arara", {})
    nano_path = RESULTS / "nanoe5" / "leaderboard_comparison.json"
    tasks = [("FaQuADIR", "FaQuADIR"), ("BRTaxQAR", "BRTaxQAR"), ("JurisTCU", "JurisTCU"),
             ("Quati", "Quati"), ("FaqBacenRetrieval", "FaqBacen")]
    rows = []
    for task, label in tasks:
        models = lb.get(task, {})
        mine = ours.get(task, {})
        if not models or not mine:
            continue
        best = max(mine.values())
        rows.append((task, label, sorted(models.values()), best))
    if not rows:
        return

    nano = {}
    if nano_path.exists():
        nano_blob = json.loads(nano_path.read_text()).get("arara", {})
        for task, exps in nano_blob.items():
            nano[task] = max(exps.values()) if exps else None

    fig, ax = plt.subplots(figsize=(9.8, 4.8), dpi=200)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    for i, (task, label, scores, mine) in enumerate(rows):
        y = len(rows) - i - 1
        ax.scatter(scores, [y] * len(scores), s=18, color=NEUTRAL, alpha=0.85,
                   zorder=2, label="MTEB-BR leaderboard models" if i == 0 else None)
        ax.scatter([mine], [y], s=105, color=ACCENT_SOFT, zorder=4, marker="D",
                   label="arara-rag, static (best mode)" if i == 0 else None)
        beaten = sum(1 for s in scores if s < mine)
        ax.text(mine, y + 0.30, f"{beaten}/{len(scores)}", ha="center", fontsize=9,
                color=MUTED, fontweight="bold")
        nv = nano.get(task)
        if nv:
            ax.scatter([nv], [y], s=125, color=ACCENT, zorder=5, marker="D",
                       label="arara-rag, nanoE5 (best mode)" if i == 0 else None)
            nb = sum(1 for s in scores if s < nv)
            ax.text(nv, y - 0.42, f"{nb}/{len(scores)}", ha="center", fontsize=9,
                    color=ACCENT, fontweight="bold")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[1] for r in rows][::-1], color=INK, fontsize=10)
    ax.set_xlabel("main metric (nDCG@10)", color=MUTED, fontsize=10)
    ax.set_title("A 200 MB CPU stack against 96 published embedding models",
                 fontsize=15, color=INK, fontweight="bold", loc="left", pad=26)
    ax.text(0, 1.05, "each grey dot is a leaderboard model; labels count how many each backend beats",
            transform=ax.transAxes, fontsize=10, color=MUTED)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="upper left")
    fig.tight_layout()
    fig.savefig(DOCS / "leaderboard.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    ablation = _load("summary_brtaxqa.json")
    nano_ablation = _load("nanoe5/summary_brtaxqa.json")
    if ablation:
        chart_ablation(
            {r["experiment"]: r for r in ablation},
            {r["experiment"]: r for r in (nano_ablation or [])},
        )
        print("wrote docs/ablation.png")
    core, nano_core = _load("summary_core.json"), _load("nanoe5/summary_core.json")
    if core and nano_core:
        chart_encoder_gain({r["experiment"]: r for r in core},
                           {r["experiment"]: r for r in nano_core})
        print("wrote docs/encoders.png")
    profile = _load("profile.json")
    if profile:
        chart_profile(profile, _load("profile_reopen.json"))
        print("wrote docs/scaling.png")
    chart_leaderboard()
    if (RESULTS / "leaderboard_comparison.json").exists():
        print("wrote docs/leaderboard.png")


if __name__ == "__main__":
    main()
