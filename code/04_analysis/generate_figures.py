#!/usr/bin/env python3
"""
04_analysis/generate_figures.py — v4 (paper-alignment fixes)

CHANGES vs v3:
  - FIG 1 FIX (was a blocker for the paper's Exp1/Exp7 argument): the bar
    chart previously showed ONLY the ungated HalluScore, with CheXagent
    ranked #1 (MCC≈0.03, near-degenerate) and no visual indication that
    this ranking is corrected elsewhere in the paper. fig1_halluscore_bar()
    now renders TWO panels side-by-side: (left) the original ungated
    ranking, clearly labeled "PRE-GATE", and (right) the MCC-gated ranking
    (floor=0.05, per Exp 7 / Table 5) with CheXagent's Discrimination
    zeroed out and MedGemma promoted to #1. A reader who only looks at
    Fig 1 now sees the correction in the same figure, rather than the
    pre-gate ranking standing alone as if it were the paper's conclusion.
  - FIG 7 FIX (was a blocker: the per-type bars for Qwen/LLaVA/Phi-3/
    CheXagent are exactly the same near-100%/near-0% degenerate pattern
    already excluded from Table 4 with an explicit footnote explaining why
    per-type recall is uninformative for near-constant classifiers;
    including the full 8-model bar chart without that context directly
    contradicts the footnote's point). fig7_per_type_accuracy() now:
      (a) visually flags degenerate models (recall ~100% or ~0% on ALL
          three types, matching the near-constant-classifier pattern from
          Exp 2) with a hatched bar style and a "(near-constant)" suffix
          on the x-axis label, and
      (b) adds an explanatory annotation box on each subplot stating that
          hatched bars reflect the collapse pattern documented in Exp 2 /
          Table 6, not genuine per-type detection ability.
    This makes the figure corroborate the paper's degeneracy claim
    instead of silently restating the exact numbers Table 4's footnote
    told the reader to discount.
  - FIG 3 (calibration) captions now explicitly flag which models' flat/
    near-2-point reliability curves are additional visual evidence of the
    Exp 2 near-constant-classifier collapse (Qwen2.5-VL, LLaVA-1.5,
    Phi-3-Vision, CheXagent), tying the figure to the paper's argument
    rather than leaving it as an uncaptioned calibration diagram.
  - FIG 5 (confidence distributions) unchanged in plotting logic (this was
    already the strongest piece of visual evidence for the GT-tracking-gap
    finding — μ_faithful vs μ_hallu separation is stark for MedGemma and
    near-zero for Qwen/CheXagent) but the per-model annotation box now
    also prints the GT-tracking gap (μ_hallu − μ_faithful) directly on the
    plot, so the figure is self-contained evidence for Table 6's claim
    without requiring the reader to cross-reference.
  - FIG 6 (shuffled-image) caption updated to explicitly reference the
    McNemar significance result (Qwen: not significant, p=0.118; MedGemma:
    significant, p<0.0001) rather than leaving "small gap" as a purely
    visual, uncited judgment call.

Everything else (Fig 2 ROC curves, Fig 4 confusion matrices) is unchanged
from v3 — those were already flagged as clean/no-conflict.
"""

import json
from pathlib import Path

import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, f1_score

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[3]
CFG_PATH = ROOT / "code/v1/config.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

OUTPUTS = ROOT / CFG["paths"]["outputs"]
METRICS = ROOT / CFG["paths"]["metrics"]
FIGURES = ROOT / CFG["paths"]["figures"]
FIGURES.mkdir(parents=True, exist_ok=True)

OUTPUT_VERSION = "v3"

# Read from config so this can never silently disagree with the binning
# used to compute the ECE numbers shown alongside these figures.
N_BINS = CFG["metrics"]["ece_bins"]

# MCC floor for the gated Discrimination fix, matching Exp 1 / Exp 7 /
# Table 5 in the paper. Kept as a named constant here (not re-derived) so
# Fig 1's gated panel can never silently drift from the value reported in
# the paper's text and tables.
MCC_GATE_FLOOR = 0.05

VLM_MODEL_KEYS = ["qwen25_vl", "medgemma", "llava15", "phi3_vision",
                  "chexagent", "biovil_t", "biomedclip", "blip2"]
BASELINE_KEYS  = ["constant_faithful", "random", "text_only_baseline"]
SHUFFLED_BASELINE_KEYS = [
    "shuffled_image_baseline_qwen25_vl",
    "shuffled_image_baseline_medgemma",
]

# Models whose per-type / per-model behavior matches the near-constant-
# classifier collapse documented in Exp 2 (Table 6): predicted-hallucinated
# rate ~100% or ~0% regardless of ground truth, GT-tracking gap ~0.
# Used to visually flag degenerate models in Fig 1 (right panel) and Fig 7
# (hatched bars) rather than silently mixing them with genuinely
# discriminative models.
DEGENERATE_MODEL_KEYS = {"chexagent", "qwen25_vl", "llava15", "phi3_vision"}

MODEL_LABELS = {
    "qwen25_vl":    "Qwen2.5-VL-7B",
    "medgemma":     "MedGemma-27B",
    "llava15":      "LLaVA-1.5-13B",
    "phi3_vision":  "Phi-3-Vision-4B",
    "chexagent":    "CheXagent-8B",
    "biovil_t":     "BioViL-T",
    "biomedclip":   "BiomedCLIP-300M",
    "blip2":        "BLIP-2-OPT-6.7B",
    "constant_faithful":                "Constant (always faithful)",
    "random":                           "Random",
    "text_only_baseline":               "Text-only heuristic",
    "shuffled_image_baseline_qwen25_vl":"Shuffled-image (Qwen2.5-VL)",
    "shuffled_image_baseline_medgemma": "Shuffled-image (MedGemma)",
}
MODEL_COLORS = {
    "qwen25_vl":    "#4C72B0",
    "medgemma":     "#DD8452",
    "llava15":      "#55A868",
    "phi3_vision":  "#C44E52",
    "chexagent":    "#8172B2",
    "biovil_t":     "#937860",
    "biomedclip":   "#DA8BC3",
    "blip2":        "#8C8C8C",
    "constant_faithful":                "#BBBBBB",
    "random":                           "#999999",
    "text_only_baseline":               "#777777",
    "shuffled_image_baseline_qwen25_vl":"#8FA9CC",
    "shuffled_image_baseline_medgemma": "#EFC499",
}

# McNemar p-values for the shuffled-image validity check, taken directly
# from significance_tests.json, so Fig 6's caption can cite the actual
# statistical result rather than an unsupported visual "small gap" claim.
SHUFFLED_IMAGE_MCNEMAR_P = {
    "qwen25_vl": 0.1183,   # qwen25_vl vs shuffled_image_baseline_qwen25_vl: NOT significant
    "medgemma":  0.0000,   # medgemma vs shuffled_image_baseline_medgemma: significant (p<0.0001)
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

SAVE_KWARGS = dict(dpi=300, bbox_inches="tight")


def savefig(fig, stem: str):
    for ext in ("png", "pdf"):
        out = FIGURES / f"{stem}.{ext}"
        fig.savefig(out, **SAVE_KWARGS)
    print(f"  → {FIGURES / stem}.png / .pdf")
    plt.close(fig)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_metrics(m):
    with open(METRICS / f"{m}_metrics.json") as f:
        return json.load(f)

def load_outputs(m):
    path = OUTPUTS / f"{m}_{OUTPUT_VERSION}.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def get_arrays(records):
    clean = [r for r in records
             if r["pred_label"] in ("faithful", "hallucinated")
             and r.get("pred_prob_hallucinated") is not None]
    y_true = np.array([1 if r["gt_hallucinated"] else 0 for r in clean])
    y_pred = np.array([1 if r["pred_label"] == "hallucinated" else 0 for r in clean])
    y_conf = np.array([r["pred_prob_hallucinated"] for r in clean])
    return y_true, y_pred, y_conf


def gated_discrimination(mt: dict, floor: float = MCC_GATE_FLOOR) -> float:
    """Applies the Exp 1 / Exp 7 MCC-gated fix to a single model's metrics
    dict: zero out Discrimination whenever MCC < floor, else pass through
    the original value. Mirrors compute_metrics.py's discrimination_fn
    exactly, just gated -- kept local here (not imported) so this figure
    script has no hidden runtime dependency on compute_metrics.py's
    internals beyond the metrics JSON contract it already reads."""
    mcc = mt.get("mcc", 0.0)
    disc = mt.get("discrimination", 0.0)
    return 0.0 if mcc < floor else disc


def gated_hallu_score(mt: dict, floor: float = MCC_GATE_FLOOR,
                       w_disc: float = 0.5, w_ece: float = 0.3, w_type: float = 0.2) -> float:
    disc_gated = gated_discrimination(mt, floor)
    ece = mt.get("ece", 0.0)
    type_acc = mt.get("type_accuracy", {}).get("macro_avg", 0.0)
    return w_disc * disc_gated + w_ece * (1 - ece) + w_type * type_acc


# ── Fig 1: HalluScore bar — NOW TWO PANELS (pre-gate vs. MCC-gated) ──────────

def fig1_halluscore_bar(all_metrics, model_keys):
    models = [m for m in model_keys if m in all_metrics]

    scores_pre  = [all_metrics[m]["hallu_score"] for m in models]
    scores_gated = [gated_hallu_score(all_metrics[m]) for m in models]
    labels_base = [MODEL_LABELS[m] for m in models]
    colors_base = [MODEL_COLORS[m] for m in models]
    is_degenerate = [m in DEGENERATE_MODEL_KEYS for m in models]

    fig, (ax_pre, ax_gated) = plt.subplots(1, 2, figsize=(13, 4.8))

    for ax, scores, title in [
        (ax_pre, scores_pre, "PRE-GATE (original formula)\nas reported in Table 3"),
        (ax_gated, scores_gated, f"MCC-GATED (floor={MCC_GATE_FLOOR})\ncorrected ranking, per Table 5"),
    ]:
        order = np.argsort(scores)[::-1]
        o_scores = [scores[i] for i in order]
        o_labels = [labels_base[i] for i in order]
        o_colors = [colors_base[i] for i in order]
        o_degen  = [is_degenerate[i] for i in order]

        bars = ax.barh(o_labels, o_scores, color=o_colors, height=0.55, edgecolor="white")
        for bar, score, degen in zip(bars, o_scores, o_degen):
            if degen:
                bar.set_hatch("///")
                bar.set_edgecolor("#555555")
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{score:.4f}", va="center", ha="left", fontweight="bold", fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("HalluScore  (↑ better)")
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.invert_yaxis()

    legend_handles = [
        Patch(facecolor="white", edgecolor="#555555", hatch="///",
              label="Near-constant classifier\n(Exp 2 collapse pattern)"),
    ]
    ax_pre.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.9)

    legend_text = ("HalluScore = 0.5 × Discrimination\n"
                   "           + 0.3 × (1 − ECE)\n"
                   "           + 0.2 × TypeAccuracy\n"
                   "Discrimination = √(MCC_norm × F1)\n"
                   f"Gated: Discrimination→0 if MCC<{MCC_GATE_FLOOR}")
    ax_gated.text(0.98, 0.03, legend_text, transform=ax_gated.transAxes,
                  ha="right", va="bottom", fontsize=7.5,
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", alpha=0.85))

    fig.suptitle("SAFER Benchmark — HalluScore, Pre-Gate vs. MCC-Gated\n"
                  "(hatched bars = near-constant classifier per Exp 2; see Table 6)",
                  fontweight="bold", fontsize=13, y=1.04)
    plt.tight_layout()
    savefig(fig, "fig1_halluscore_bar")


# ── Fig 2: ROC curves ─────────────────────────────────────────────────────────
# Unchanged from v3 — no auto-inversion, clean AUROC values matching Table 3.

def fig2_roc_curves(all_outputs, model_keys):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    legend_handles = []
    for m in model_keys:
        if m not in all_outputs or not all_outputs[m]:
            continue
        y_true, _, y_conf = get_arrays(all_outputs[m])
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_conf)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=MODEL_COLORS[m], linewidth=2.2)
        legend_handles.append(
            Line2D([0], [0], color=MODEL_COLORS[m], linewidth=2.5,
                   label=f"{MODEL_LABELS[m]}  (AUC = {roc_auc:.3f})")
        )

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    legend_handles.append(
        Line2D([0], [0], color="gray", linestyle="--",
               linewidth=1.2, label="Random  (AUC = 0.500)")
    )

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Hallucination Detection\n"
                 "(P(hallucinated) elicited directly; no post-hoc inversion)",
                 fontweight="bold", fontsize=13, pad=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.legend(handles=legend_handles, loc="best", framealpha=0.95,
              edgecolor="#cccccc", fontsize=10, handlelength=2.2, labelspacing=0.5)
    plt.tight_layout()
    savefig(fig, "fig2_roc_curves")


# ── Fig 3: Calibration — captions now flag degenerate-model flat curves ─────

def fig3_calibration(all_outputs, all_metrics, model_keys):
    valid_keys = [m for m in model_keys if m in all_outputs and all_outputs[m]]
    pairs = [(valid_keys[i], valid_keys[i+1] if i+1 < len(valid_keys) else None)
             for i in range(0, len(valid_keys), 2)]

    for fig_idx, (m1, m2) in enumerate(pairs, start=1):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        any_degenerate_in_pair = False
        for ax, m in zip(axes, [m1, m2]):
            if m is None or m not in all_outputs or not all_outputs[m]:
                ax.axis("off"); continue
            y_true, _, y_conf = get_arrays(all_outputs[m])
            if len(y_true) == 0:
                ax.axis("off"); continue
            bin_edges = np.linspace(0, 1, N_BINS + 1)
            bin_accs, bin_confs = [], []
            for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
                mask = (y_conf >= lo) & (y_conf <= hi if hi == 1.0 else y_conf < hi)
                if mask.sum() == 0: continue
                bin_accs.append(y_true[mask].mean())
                bin_confs.append(y_conf[mask].mean())
            bin_accs, bin_confs = np.array(bin_accs), np.array(bin_confs)
            ax.plot([0,1],[0,1],"k--",linewidth=0.8,alpha=0.5,label="Perfect")
            bars = ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.55,
                          color=MODEL_COLORS[m], label="Model")
            ax.plot(bin_confs, bin_accs, "o-", color=MODEL_COLORS[m],
                    linewidth=1.5, markersize=5)
            for bar, acc in zip(bars, bin_accs):
                if acc < 0.02: continue
                label_y = bar.get_height() + 0.03
                va, color = "bottom", MODEL_COLORS[m]
                if label_y > 1.08:
                    label_y = bar.get_height() - 0.06
                    va, color = "top", "white"
                ax.text(bar.get_x() + bar.get_width()/2, label_y, f"{acc:.2f}",
                        ha="center", va=va, fontsize=7.5, fontweight="bold", color=color)
            ece_val = all_metrics.get(m, {}).get("ece")
            title = MODEL_LABELS[m]
            if m in DEGENERATE_MODEL_KEYS:
                title += "  ⚠"
                any_degenerate_in_pair = True
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Mean Predicted P(hallucinated)")
            ax.set_ylabel("Fraction Hallucinated")
            ax.set_xlim(0,1); ax.set_ylim(0,1.12)
            if ece_val is not None:
                ax.text(0.95, 0.92, f"ECE = {ece_val:.4f}", transform=ax.transAxes,
                        fontsize=9, color=MODEL_COLORS[m], fontweight="bold", ha="right")
            ax.legend(fontsize=9)

        subtitle = "Calibration / Reliability Diagrams"
        if any_degenerate_in_pair:
            subtitle += ("\n⚠ = near-constant classifier (Exp 2); flat/2-point curve "
                          "reflects collapsed output, not genuine calibration")
        fig.suptitle(subtitle, fontweight="bold", fontsize=13, y=1.04)
        plt.tight_layout()
        savefig(fig, f"fig3_calibration_part{fig_idx}")


# ── Fig 4: Confusion matrices ─────────────────────────────────────────────────
# Unchanged from v3.

def fig4_confusion_matrices(all_metrics, model_keys):
    valid_keys = [m for m in model_keys if m in all_metrics]
    pairs = [(valid_keys[i], valid_keys[i+1] if i+1 < len(valid_keys) else None)
             for i in range(0, len(valid_keys), 2)]

    for fig_idx, (m1, m2) in enumerate(pairs, start=1):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
        for ax, m in zip(axes, [m1, m2]):
            if m is None:
                ax.axis("off"); continue
            mt = all_metrics[m]
            cm = np.array([[mt["TN"], mt["FP"]], [mt["FN"], mt["TP"]]])
            total = cm.sum()
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(total//2, 1))
            for i in range(2):
                for j in range(2):
                    val = cm[i, j]
                    pct = val / total * 100 if total else 0
                    ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center",
                            color="white" if val > total//4 else "black",
                            fontsize=10, fontweight="bold")
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(["Pred\nFaithful", "Pred\nHallu"])
            ax.set_yticklabels(["GT\nFaithful", "GT\nHallu"])
            ax.set_title(MODEL_LABELS[m], fontweight="bold", fontsize=11)
        fig.suptitle("Confusion Matrices", fontweight="bold", fontsize=13, y=1.02)
        plt.tight_layout()
        savefig(fig, f"fig4_confusion_matrices_part{fig_idx}")


# ── Fig 5: Confidence distributions — now prints GT-tracking gap directly ───

CONF_COLORS = {
    "qwen25_vl":   {"faithful": "#7BA7D4", "hallu": "#1A4A8A"},
    "medgemma":    {"faithful": "#F5B87A", "hallu": "#A84A0A"},
    "llava15":     {"faithful": "#8FD4A8", "hallu": "#1A6B3A"},
    "phi3_vision": {"faithful": "#E89090", "hallu": "#8B1A1A"},
    "chexagent":   {"faithful": "#C4B8E0", "hallu": "#4A3080"},
    "biovil_t":    {"faithful": "#C4A882", "hallu": "#6B4020"},
    "biomedclip":  {"faithful": "#F0B8DF", "hallu": "#903060"},
    "blip2":       {"faithful": "#C8C8C8", "hallu": "#404040"},
}

def fig5_confidence_dist(all_outputs, model_keys):
    valid_keys = [m for m in model_keys if m in CONF_COLORS and all_outputs.get(m)]
    pairs = [(valid_keys[i], valid_keys[i+1] if i+1 < len(valid_keys) else None)
             for i in range(0, len(valid_keys), 2)]
    bins = np.linspace(0, 1, 21)

    for fig_idx, (m1, m2) in enumerate(pairs, start=1):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, m in zip(axes, [m1, m2]):
            if m is None:
                ax.axis("off"); continue
            records = all_outputs[m]
            clean = [r for r in records
                     if r["pred_label"] in ("faithful","hallucinated")
                     and r.get("pred_prob_hallucinated") is not None]
            conf_faith = [r["pred_prob_hallucinated"] for r in clean if not r["gt_hallucinated"]]
            conf_hallu = [r["pred_prob_hallucinated"] for r in clean if r["gt_hallucinated"]]
            cf = CONF_COLORS[m]
            ax.hist(conf_faith, bins=bins, alpha=0.75, color=cf["faithful"],
                    label="GT Faithful", edgecolor="white", linewidth=0.5, density=True)
            ax.hist(conf_hallu, bins=bins, alpha=0.75, color=cf["hallu"],
                    label="GT Hallucinated", edgecolor="white", linewidth=0.5, density=True)
            if conf_faith: ax.axvline(np.mean(conf_faith), color=cf["faithful"],
                                       linestyle="--", linewidth=1.5)
            if conf_hallu: ax.axvline(np.mean(conf_hallu), color=cf["hallu"],
                                       linestyle="--", linewidth=1.5)
            ax.set_title(MODEL_LABELS[m], fontweight="bold", fontsize=12,
                         color=MODEL_COLORS[m])
            ax.set_xlabel("P(hallucinated)", fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.set_xlim(0,1)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
            mean_f = np.mean(conf_faith) if conf_faith else 0
            mean_h = np.mean(conf_hallu) if conf_hallu else 0
            gap = mean_h - mean_f
            ax.text(0.97, 0.95,
                    f"μ faithful = {mean_f:.2f}\nμ hallu      = {mean_h:.2f}\n"
                    f"gap (μ_hallu−μ_faith) = {gap:+.2f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                              alpha=0.85, edgecolor="#dddddd"))
            ax.text(0.03, 0.95, f"n faithful={len(conf_faith)}\nn hallu={len(conf_hallu)}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#555555")
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.0),
                      framealpha=0.92, edgecolor="#cccccc", fontsize=9, ncol=2)
        fig.suptitle("Confidence Score Distributions by Ground Truth\n"
                     "(P(hallucinated), fixed direction; gap ≈ GT-tracking signal, cf. Table 6)",
                     fontweight="bold", fontsize=13, y=1.03)
        plt.tight_layout(h_pad=3.0, w_pad=2.5)
        savefig(fig, f"fig5_confidence_dist_part{fig_idx}")


# ── Fig 6: Real-image vs. shuffled-image accuracy — caption cites McNemar ──

def fig6_shuffled_image_comparison(all_metrics, real_keys=("qwen25_vl", "medgemma")):
    shuffled_key_map = {
        "qwen25_vl": "shuffled_image_baseline_qwen25_vl",
        "medgemma":  "shuffled_image_baseline_medgemma",
    }
    labels, real_accs, shuffled_accs, p_values = [], [], [], []
    for rk in real_keys:
        sk = shuffled_key_map.get(rk)
        if rk not in all_metrics or sk not in all_metrics:
            continue
        labels.append(MODEL_LABELS[rk])
        real_accs.append(all_metrics[rk]["accuracy"])
        shuffled_accs.append(all_metrics[sk]["accuracy"])
        p_values.append(SHUFFLED_IMAGE_MCNEMAR_P.get(rk))

    if not labels:
        print("  [skip] fig6_shuffled_image_comparison: no matching "
              "real/shuffled metric pairs found — run the shuffled-image "
              "baseline for at least one VLM first.")
        return

    x = np.arange(len(labels))
    width = 0.32
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bars_real = ax.bar(x - width / 2, real_accs, width, label="Real image",
                        color="#4C72B0", edgecolor="white")
    bars_shuf = ax.bar(x + width / 2, shuffled_accs, width, label="Shuffled (wrong) image",
                        color="#BBBBBB", edgecolor="white", hatch="///")

    for bars in (bars_real, bars_shuf):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    # Annotate McNemar significance directly above each model's pair of bars.
    for i, (xi, p) in enumerate(zip(x, p_values)):
        if p is None:
            continue
        sig_str = f"McNemar p={p:.4f}" + ("  (n.s.)" if p >= 0.05 else "  (sig.)")
        bar_top = max(real_accs[i], shuffled_accs[i])
        ax.text(xi, bar_top + 0.06,
                sig_str, ha="center", va="bottom", fontsize=8.5, style="italic",
                color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.15)
    ax.set_title("Real-Image vs. Shuffled-Image Accuracy\n"
                 "(McNemar test on paired predictions; n.s. = model's behavior is not\n"
                 "statistically distinguishable between real and shuffled image)",
                 fontweight="bold", pad=12, fontsize=11)
    ax.legend(loc="upper left", framealpha=0.92, edgecolor="#cccccc")
    plt.tight_layout()
    savefig(fig, "fig6_shuffled_image_comparison")


# ── Fig 7: Per-type accuracy — degenerate models now visually flagged ──────

def fig7_per_type_accuracy(all_metrics, model_keys):
    """
    Per-type hallucination detection accuracy — one figure per hallucination
    type, PLUS an overview figure with all three types side-by-side.

    FIX: models matching the Exp 2 near-constant-classifier pattern
    (DEGENERATE_MODEL_KEYS) are rendered with a hatched bar style and an
    explanatory annotation, so this figure corroborates the degeneracy
    finding (as Table 4's footnote already states) instead of presenting
    ~100%/~0% per-type "accuracy" for those models as if it reflected
    genuine type-level detection ability.
    """
    valid_keys = [m for m in model_keys if m in all_metrics]
    htypes     = ["object", "attribute", "relational"]
    htype_labels = {"object": "Object", "attribute": "Attribute", "relational": "Relational"}

    for htype in htypes:
        accs, labels, colors, hatches = [], [], [], []
        for m in valid_keys:
            ta  = all_metrics[m].get("type_accuracy", {})
            acc = ta.get(htype, {}).get("acc")
            if acc is None:
                continue
            accs.append(acc)
            lbl = MODEL_LABELS[m]
            if m in DEGENERATE_MODEL_KEYS:
                lbl += "\n(near-constant)"
            labels.append(lbl)
            colors.append(MODEL_COLORS[m])
            hatches.append("///" if m in DEGENERATE_MODEL_KEYS else None)

        fig, ax = plt.subplots(figsize=(9.5, 5.0))
        x    = np.arange(len(labels))
        bars = ax.bar(x, accs, color=colors, width=0.55, edgecolor="white")
        for bar, hatch in zip(bars, hatches):
            if hatch:
                bar.set_hatch(hatch)
                bar.set_edgecolor("#444444")

        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{acc*100:.1f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.axhline(0.5, color="gray", linestyle="--",
                   linewidth=0.9, alpha=0.6, label="chance")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9.5)
        ax.set_ylabel("Detection Accuracy  (↑ better)", fontsize=11)
        ax.set_ylim(0, 1.22)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.set_title(f"Per-Type Hallucination Detection Accuracy — {htype_labels[htype]}\n"
                     f"(recall on hallucinated samples of type '{htype}' only)",
                     fontweight="bold", fontsize=13, pad=12)

        legend_handles = [
            Line2D([0], [0], color="gray", linestyle="--", label="chance"),
            Patch(facecolor="white", edgecolor="#444444", hatch="///",
                  label="Near-constant classifier (Exp 2);\n"
                        "≈100%/≈0% here reflects collapsed\n"
                        "output, not type-level sensitivity —\n"
                        "excluded from Table 4 for this reason"),
        ]
        ax.legend(handles=legend_handles, fontsize=8, loc="upper left",
                  framealpha=0.93, edgecolor="#cccccc")
        plt.tight_layout()
        savefig(fig, f"fig7_per_type_accuracy_{htype}")

    # Overview figure (all three types, grouped bars) — same degeneracy
    # flagging applied, replacing the old ungrouped fig7_per_type_accuracy
    # that mixed genuine and degenerate models with no visual distinction.
    fig, ax = plt.subplots(figsize=(11, 5.5))
    n_models = len(valid_keys)
    width = 0.25
    x_base = np.arange(len(htypes))
    for i, m in enumerate(valid_keys):
        ta = all_metrics[m].get("type_accuracy", {})
        accs = [ta.get(h, {}).get("acc") for h in htypes]
        if any(a is None for a in accs):
            continue
        offset = (i - n_models / 2) * width / (n_models / 3)
        bars = ax.bar(x_base + offset, accs, width / (n_models / 3),
                       color=MODEL_COLORS[m], edgecolor="white",
                       label=MODEL_LABELS[m] + (" (near-const.)" if m in DEGENERATE_MODEL_KEYS else ""))
        if m in DEGENERATE_MODEL_KEYS:
            for bar in bars:
                bar.set_hatch("///")
                bar.set_edgecolor("#444444")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set_xticks(x_base)
    ax.set_xticklabels([htype_labels[h] for h in htypes], fontsize=11)
    ax.set_ylabel("Detection Accuracy (↑ better)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Type Hallucination Detection Accuracy — Overview\n"
                 "(hatched = near-constant classifier per Exp 2; see Table 4 footnote)",
                 fontweight="bold", fontsize=13)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)
    plt.tight_layout()
    savefig(fig, "fig7_per_type_accuracy")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[load] metrics and outputs ...")
    all_model_keys = VLM_MODEL_KEYS + BASELINE_KEYS + SHUFFLED_BASELINE_KEYS
    all_metrics = {}
    for m in all_model_keys:
        try:
            all_metrics[m] = load_metrics(m)
        except FileNotFoundError:
            print(f"  [skip] no metrics for {m} (run compute_metrics.py first)")
    all_outputs = {m: load_outputs(m) for m in all_model_keys}

    print("[figures]")
    fig1_halluscore_bar(all_metrics, VLM_MODEL_KEYS)
    fig2_roc_curves(all_outputs, VLM_MODEL_KEYS)
    fig3_calibration(all_outputs, all_metrics, VLM_MODEL_KEYS)
    fig4_confusion_matrices(all_metrics, VLM_MODEL_KEYS)
    fig5_confidence_dist(all_outputs, VLM_MODEL_KEYS)
    fig6_shuffled_image_comparison(all_metrics)
    fig7_per_type_accuracy(all_metrics, VLM_MODEL_KEYS)

    print(f"\n[done] figures (PNG + PDF) → {FIGURES}")
    print("  Fig 1 now has TWO panels (pre-gate vs. MCC-gated) so the")
    print("  corrected ranking is visible in the same figure as the raw one.")
    print("  Fig 7 (all sub-figures + overview) now hatches near-constant")
    print("  classifiers (chexagent, qwen25_vl, llava15, phi3_vision) per")
    print("  the Exp 2 collapse finding, matching Table 4's footnote.")
    print("  Fig 3 captions flag which models' flat calibration curves")
    print("  reflect the same collapse, not genuine calibration.")
    print("  Fig 5 now prints the GT-tracking gap directly on each panel.")
    print("  Fig 6 caption cites the actual McNemar p-values from")
    print("  significance_tests.json instead of an uncited 'small gap' claim.")