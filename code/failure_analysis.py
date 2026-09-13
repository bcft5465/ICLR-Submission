#!/usr/bin/env python3
"""
failure_analysis.py — SAFER benchmark failure-analysis experiments

Runs the experiments needed to turn the post-review SAFER resubmission
into a genuine failure-analysis contribution:

  EXP 1: Discrimination-metric degenerate-classifier blind spot
         (synthetic sweep + analytic derivation + proposed fix)
  EXP 2: MedGemma-vs-rest disagreement / shortcut-hypothesis analysis
  EXP 3: Per-source (OpenI vs MIMIC-CXR) generalization check
  EXP 4: HalluScore rank-instability driver analysis + Pareto frontier
  EXP 5: Text-only leakage — trained classifier + paraphrase-robustness test
  EXP 6: Paraphrase-repair FIX for the Exp 5 leak, re-validated end-to-end
         (grammar-repairing rewrite of the SWAP_MAP injection by default;
         optional LLM-paraphrase path if an API key is present — see EXP 6
         docstring below for exactly how to enable it)
  EXP 7: Closes the loop on Exp 1 — validates the MCC-gated Discrimination
         fix against REAL model metrics (not just the synthetic sweep),
         confirming it demotes CheXagent-style near-degenerate models while
         leaving MedGemma's rank intact
  EXP 8: Closed-vocabulary hypothesis test (NO API NEEDED) — directly tests
         whether Exp 5's leak is explained by the fixed, closed SWAP_MAP
         antonym vocabulary itself (as opposed to grammar/collocation
         residue, which Exp 6 already ruled out) by excluding the ENTIRE
         swap vocabulary from the classifier's feature space and comparing
         against a size-matched random-vocabulary control
  EXP 9: Eligibility-confound test (NO API NEEDED) — Exp 8 found the swap
         vocabulary explains only ~half the leak (AUROC 0.918 -> 0.689, not
         to chance). This tests whether the residual is a REPORT-LENGTH
         confound, a BROADER-VOCABULARY confound (reports eligible for
         injection use more anatomical language in general, beyond just
         SWAP_MAP words), or a genuine structural difference between
         injection-eligible and non-eligible reports
  EXP 10: Source-imbalance leak test (NO API NEEDED) — SAFER combines OpenI
          + MIMIC-CXR; tests whether the Exp 9 residual is explained by
          differing report-writing conventions between sources rather than
          anything about the injected hallucination itself

USAGE
-----
    python failure_analysis.py --root /mnt/ssd/users/durgesh/HEalthcare/Health2

    # Run only specific experiments:
    python failure_analysis.py --root . --only exp6 exp7

    # EXP 6 optional LLM-paraphrase path (off by default, uses the
    # rule-based grammar-repair path unless you pass this flag AND have
    # ANTHROPIC_API_KEY set in your environment):
    python failure_analysis.py --root . --only exp6 --exp6_use_llm

Expects the standard SAFER directory layout under --root:
    results/v1/outputs/<model>_v3.jsonl        (raw per-sample records)
    results/v1/metrics/<model>_metrics.json    (aggregated metrics)
    results/v1/metrics/summary.csv
    data/v1/processed/benchmark_multi_v1.jsonl (benchmark definition, for source field)

Each experiment is independent: if its required input files are missing,
it prints a clear [skip] message naming the exact file it needed, rather
than crashing the whole run or silently fabricating a result. This matters
here specifically because the paper's entire point is "don't hide missing
or unreliable inputs" — this script should not violate its own thesis.

Outputs go to <root>/results/v1/failure_analysis/:
    exp1_degenerate_blindspot.png / .csv / .json
    exp2_medgemma_disagreement.csv / .json
    exp3_per_source_accuracy.csv / .json
    exp4_rank_instability.png / .csv / .json
    exp5_text_leakage.json
    exp6_paraphrase_fix_benchmark.jsonl   (the repaired benchmark, same schema)
    exp6_paraphrase_fix_validation.json   (re-run of Exp 5's classifier on it)
    exp7_mcc_gate_validation.csv / .json
    exp8_closed_vocabulary_test.json
    exp9_eligibility_confound.json
    exp10_source_imbalance.json
    failure_analysis_report.md   (human-readable summary of all experiments)
"""

import argparse
import json
import re
import warnings
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from sklearn.metrics import (
        f1_score, matthews_corrcoef, roc_auc_score, accuracy_score,
        balanced_accuracy_score, confusion_matrix,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from scipy.stats import spearmanr


# ══════════════════════════════════════════════════════════════════════════
# Shared loading helpers
# ══════════════════════════════════════════════════════════════════════════

def load_metrics_json(root: Path, model_key: str) -> dict | None:
    p = root / "results/v1/metrics" / f"{model_key}_metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_outputs_jsonl(root: Path, model_key: str, version: str = "v3") -> list[dict] | None:
    p = root / "results/v1/outputs" / f"{model_key}_{version}.jsonl"
    if not p.exists():
        return None
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_benchmark(root: Path) -> dict[str, dict] | None:
    """Returns id -> benchmark record (has 'source', 'report', 'hallu_type', etc.)"""
    for name in ("benchmark_multi_v1.jsonl", "benchmark_v1.jsonl"):
        p = root / "data/v1/processed" / name
        if p.exists():
            with open(p) as f:
                recs = [json.loads(l) for l in f if l.strip()]
            return {r["id"]: r for r in recs}
    return None


def load_summary_csv(root: Path) -> pd.DataFrame | None:
    p = root / "results/v1/metrics/summary.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def clean_records(records: list[dict]) -> list[dict]:
    """Filter to records with a usable parsed prediction."""
    return [r for r in records
            if r.get("pred_label") in ("faithful", "hallucinated")
            and r.get("pred_prob_hallucinated") is not None]


def to_arrays(records: list[dict]):
    y_true = np.array([1 if r["gt_hallucinated"] else 0 for r in records])
    y_pred = np.array([1 if r["pred_label"] == "hallucinated" else 0 for r in records])
    y_conf = np.array([r["pred_prob_hallucinated"] for r in records])
    return y_true, y_pred, y_conf


def mcc_norm_f1_discrimination(mcc: float, f1: float) -> float:
    mcc_norm = (mcc + 1.0) / 2.0
    return float(np.sqrt(max(0.0, mcc_norm) * max(0.0, f1)))


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Discrimination blind-spot: analytic sweep + proposed fix
# ══════════════════════════════════════════════════════════════════════════
#
# Question: at what recall level (with specificity forced toward 0, i.e. an
# almost-always-predict-"hallucinated" classifier) does Discrimination =
# sqrt(MCC_norm * F1) actually collapse to ~0? The original HalluScore fix
# was designed to punish LLaMA's always-FAITHFUL degeneracy; CheXagent shows
# the SAME formula does not punish an always-HALLUCINATED degeneracy nearly
# as hard, because F1 (recall-weighted) stays high in that direction while
# MCC only mildly drops. This experiment characterizes the blind spot with
# synthetic classifiers at the real class prior, not just the one real
# example (CheXagent) — a single example is an anecdote, a swept curve is a
# characterization.
#
# It also evaluates one candidate fix: gating Discrimination on a MCC floor,
# i.e. Discrimination_fixed = 0 whenever MCC < mcc_floor, else the original
# formula. This is deliberately simple (not claimed to be the "best" fix —
# that would need its own ablation) so the paper can show the blind spot is
# closable, not just diagnosable.

def exp1_degenerate_blindspot(root: Path, out_dir: Path, class_prior: float = 0.4706,
                               n_samples: int = 8294, mcc_floor: float = 0.05):
    print("\n" + "=" * 70)
    print("EXP 1: Discrimination-metric degenerate-classifier blind spot")
    print("=" * 70)

    n_pos = int(round(n_samples * class_prior))   # hallucinated
    n_neg = n_samples - n_pos                      # faithful
    rng = np.random.default_rng(42)

    y_true = np.array([1] * n_pos + [0] * n_neg)

    recall_grid = np.round(np.arange(0.50, 1.001, 0.02), 3)
    rows = []
    for recall in recall_grid:
        # Specificity forced toward 0 (mostly predicting "hallucinated"
        # regardless of truth) — this is the CheXagent/LLaMA failure shape.
        # We sweep specificity at a fixed LOW value (0.03, matching
        # CheXagent's observed 0.0317) so the only thing varying is recall,
        # isolating recall's effect on the metric.
        specificity = 0.03
        n_tp = int(round(recall * n_pos))
        n_fn = n_pos - n_tp
        n_tn = int(round(specificity * n_neg))
        n_fp = n_neg - n_tn

        y_pred = np.zeros(n_samples, dtype=int)
        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]
        tp_idx = rng.choice(pos_idx, size=n_tp, replace=False)
        tn_idx = rng.choice(neg_idx, size=n_tn, replace=False)
        y_pred[tp_idx] = 1
        y_pred[np.setdiff1d(pos_idx, tp_idx)] = 0
        y_pred[np.setdiff1d(neg_idx, tn_idx)] = 1
        y_pred[tn_idx] = 0

        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
        disc = mcc_norm_f1_discrimination(mcc, f1)
        disc_fixed = 0.0 if mcc < mcc_floor else disc

        rows.append({
            "recall": recall, "specificity_fixed_at": specificity,
            "f1": round(f1, 4), "mcc": round(mcc, 4),
            "discrimination_original": round(disc, 4),
            "discrimination_mcc_gated": round(disc_fixed, 4),
        })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "exp1_degenerate_blindspot.csv"
    df.to_csv(csv_path, index=False)
    print(f"  → {csv_path}")

    # Locate real models on this curve for direct comparison
    real_points = []
    for model_key in ["chexagent", "qwen25_vl", "llava15", "phi3_vision",
                       "medgemma", "biovil_t", "biomedclip", "blip2",
                       "constant_faithful"]:
        m = load_metrics_json(root, model_key)
        if m is None:
            continue
        real_points.append({
            "model": model_key, "recall": m["recall"], "specificity": m["specificity"],
            "mcc": m["mcc"], "f1": m["f1"], "discrimination": m["discrimination"],
        })
    real_df = pd.DataFrame(real_points)
    real_path = out_dir / "exp1_real_model_positions.csv"
    real_df.to_csv(real_path, index=False)
    print(f"  → {real_path}")

    # Where does the ORIGINAL formula cross below 0.1 (near-zero) vs. the
    # MCC-gated fix? This is the headline number for the paper.
    thresh = 0.10
    orig_cross = df[df["discrimination_original"] < thresh]["recall"].max()
    fixed_cross_recalls = df[df["discrimination_mcc_gated"] >= thresh]["recall"]
    summary = {
        "class_prior_hallucinated": class_prior,
        "specificity_fixed_at": 0.03,
        "mcc_floor_tested": mcc_floor,
        "highest_recall_where_original_disc_below_0.10": (
            float(orig_cross) if not pd.isna(orig_cross) else None
        ),
        "note": (
            "If the ORIGINAL discrimination formula only drops below 0.10 at "
            "very high recall (close to 1.0), that IS the blind spot: a model "
            "predicting 'hallucinated' ~90-97% of the time (like CheXagent, "
            "recall=0.979) still scores well above the near-degenerate "
            "threshold. The MCC-gated fix should show a sharper, earlier cutoff."
        ),
        "real_model_discrimination_at_reported_recall": real_df.to_dict(orient="records"),
    }
    json_path = out_dir / "exp1_degenerate_blindspot.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  → {json_path}")

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df["recall"], df["discrimination_original"], "o-",
                label="Discrimination (original) = √(MCC_norm × F1)", color="#C44E52")
        ax.plot(df["recall"], df["discrimination_mcc_gated"], "s--",
                label=f"Discrimination (MCC-gated fix, floor={mcc_floor})", color="#4C72B0")
        ax.axhline(0.10, color="gray", linestyle=":", linewidth=1, alpha=0.7,
                    label="near-degenerate threshold (0.10)")
        # Overlay real models by their recall
        for _, r in real_df.iterrows():
            ax.scatter([r["recall"]], [r["discrimination"]], marker="*", s=180,
                       zorder=5, edgecolor="black")
            ax.annotate(r["model"], (r["recall"], r["discrimination"]),
                        textcoords="offset points", xytext=(5, 6), fontsize=8)
        ax.set_xlabel("Recall (with specificity fixed at 0.03, mimicking observed\n"
                       "near-degenerate 'always predict hallucinated' models)")
        ax.set_ylabel("Discrimination score")
        ax.set_title("EXP 1: Degenerate-classifier blind spot in HalluScore's\n"
                      "Discrimination term, synthetic sweep vs. real models",
                      fontweight="bold")
        ax.legend(fontsize=8.5, loc="upper left")
        plt.tight_layout()
        fig_path = out_dir / "exp1_degenerate_blindspot.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {fig_path}")
    else:
        print("  [note] matplotlib not available — skipped PNG, CSV/JSON still produced")

    return summary


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — MedGemma-vs-rest disagreement / shortcut hypothesis
# ══════════════════════════════════════════════════════════════════════════
#
# cross_model_verification.json shows medgemma agrees with
# qwen25_vl/llava15/phi3_vision/chexagent only ~38% of the time, while THOSE
# FOUR agree with each other >97% of the time. This experiment tests whether
# the qwen/llava/phi3/chexagent cluster is exploiting a SHARED superficial
# cue (report length, presence of swap vocabulary, position of the injected
# span) that medgemma does not rely on — i.e. is medgemma actually different,
# or just differently-biased?
#
# REQUIRES: raw per-sample outputs (*_v3.jsonl) for medgemma + at least one
# cluster model, AND the benchmark file (for report text / hallu_type /
# original_span / injected_span). If these aren't present, prints exactly
# which file is missing rather than approximating with the metrics JSON
# alone (which has no per-sample text).

def exp2_medgemma_disagreement(root: Path, out_dir: Path,
                                cluster_models=("qwen25_vl", "llava15", "phi3_vision", "chexagent")):
    print("\n" + "=" * 70)
    print("EXP 2: MedGemma-vs-cluster disagreement / shortcut hypothesis")
    print("=" * 70)

    bench = load_benchmark(root)
    if bench is None:
        print("  [skip] benchmark_multi_v1.jsonl / benchmark_v1.jsonl not found under "
              "data/v1/processed/ — needed for report text / span info. "
              "Re-run with --root pointing at the pipeline root.")
        return None

    medgemma_recs = load_outputs_jsonl(root, "medgemma")
    if medgemma_recs is None:
        print("  [skip] results/v1/outputs/medgemma_v3.jsonl not found. This "
              "experiment needs RAW per-sample outputs, not just the aggregated "
              "metrics JSON, because it inspects individual predictions.")
        return None

    medgemma_by_id = {r["id"]: r for r in clean_records(medgemma_recs)}

    cluster_by_model = {}
    for mk in cluster_models:
        recs = load_outputs_jsonl(root, mk)
        if recs is None:
            print(f"  [skip-model] {mk}_v3.jsonl not found — excluding from cluster")
            continue
        cluster_by_model[mk] = {r["id"]: r for r in clean_records(recs)}

    if not cluster_by_model:
        print("  [skip] none of the cluster models' output files were found")
        return None

    common_ids = set(medgemma_by_id)
    for mk, d in cluster_by_model.items():
        common_ids &= set(d)
    common_ids = sorted(common_ids)
    print(f"  {len(common_ids)} samples common to medgemma + {list(cluster_by_model)}")

    rows = []
    for sid in common_ids:
        bench_rec = bench.get(sid, {})
        mg_pred = medgemma_by_id[sid]["pred_label"]

        cluster_preds = [cluster_by_model[mk][sid]["pred_label"] for mk in cluster_by_model]
        cluster_majority = Counter(cluster_preds).most_common(1)[0][0]
        cluster_agrees_internally = len(set(cluster_preds)) == 1

        report_text = bench_rec.get("report", "")
        rows.append({
            "id": sid,
            "gt_hallucinated": bench_rec.get("hallucinated"),
            "hallu_type": bench_rec.get("hallu_type"),
            "medgemma_pred": mg_pred,
            "cluster_majority_pred": cluster_majority,
            "cluster_internally_agrees": cluster_agrees_internally,
            "medgemma_agrees_with_cluster": (mg_pred == cluster_majority),
            "report_len_chars": len(report_text),
            "report_len_words": len(report_text.split()),
            "injected_span": bench_rec.get("injected_span"),
            "original_span": bench_rec.get("original_span"),
        })

    df = pd.DataFrame(rows)
    disagree_df = df[~df["medgemma_agrees_with_cluster"]]
    agree_df = df[df["medgemma_agrees_with_cluster"]]

    print(f"  Overall agreement rate: {df['medgemma_agrees_with_cluster'].mean():.4f}")
    print(f"  Disagreement set size: {len(disagree_df)}")

    # ── Shortcut hypothesis test 1: report length ──
    len_stats = {
        "agree_mean_words": float(agree_df["report_len_words"].mean()) if len(agree_df) else None,
        "disagree_mean_words": float(disagree_df["report_len_words"].mean()) if len(disagree_df) else None,
    }

    # ── Shortcut hypothesis test 2: does cluster majority correlate with
    #    predicting "hallucinated" more often on LONGER reports (a plausible
    #    surface shortcut: longer report -> more chances something looks off)? ──
    cluster_hallu_mask = df["cluster_majority_pred"] == "hallucinated"
    if cluster_hallu_mask.sum() > 5 and (~cluster_hallu_mask).sum() > 5:
        len_by_cluster_verdict = {
            "cluster_says_hallucinated_mean_words": float(df.loc[cluster_hallu_mask, "report_len_words"].mean()),
            "cluster_says_faithful_mean_words": float(df.loc[~cluster_hallu_mask, "report_len_words"].mean()),
        }
        rho_len_vs_clusterpred, p_len_vs_clusterpred = spearmanr(
            df["report_len_words"], cluster_hallu_mask.astype(int)
        )
    else:
        len_by_cluster_verdict = {}
        rho_len_vs_clusterpred, p_len_vs_clusterpred = None, None

    # ── Shortcut hypothesis test 3: does the cluster's "hallucinated" call
    #    correlate simply with GT being hallucinated at all (i.e. are they
    #    tracking ground truth, or a fixed base-rate skew)? Compare cluster
    #    majority's predicted-hallucinated RATE against medgemma's, split by
    #    actual GT. A shortcut model's rate barely moves between GT classes;
    #    a grounded model's rate should differ sharply. ──
    def rate_by_gt(pred_col):
        out = {}
        for gt_val, label in [(True, "gt_hallucinated"), (False, "gt_faithful")]:
            sub = df[df["gt_hallucinated"] == gt_val]
            if len(sub) == 0:
                out[label] = None
                continue
            out[label] = float((sub[pred_col] == "hallucinated").mean())
        return out

    rate_analysis = {
        "medgemma_pred_hallucinated_rate": rate_by_gt("medgemma_pred"),
        "cluster_majority_pred_hallucinated_rate": rate_by_gt("cluster_majority_pred"),
    }

    # ── Per-hallu-type breakdown of disagreement ──
    disagree_by_type = (
        disagree_df.groupby("hallu_type", dropna=False).size().to_dict()
        if len(disagree_df) else {}
    )
    total_by_type = df.groupby("hallu_type", dropna=False).size().to_dict()
    disagree_rate_by_type = {
        k: round(disagree_by_type.get(k, 0) / v, 4) for k, v in total_by_type.items()
    }

    result = {
        "n_common_samples": len(common_ids),
        "cluster_models": list(cluster_by_model),
        "overall_medgemma_cluster_agreement": round(float(df["medgemma_agrees_with_cluster"].mean()), 4),
        "report_length_shortcut_check": {
            **len_stats,
            "interpretation": (
                "If disagree_mean_words differs substantially from "
                "agree_mean_words, report length may partly explain WHERE "
                "medgemma and the cluster diverge (not necessarily WHY either "
                "is right)."
            ),
        },
        "length_vs_cluster_verdict": {
            **len_by_cluster_verdict,
            "spearman_rho": round(float(rho_len_vs_clusterpred), 4) if rho_len_vs_clusterpred is not None else None,
            "p_value": round(float(p_len_vs_clusterpred), 4) if p_len_vs_clusterpred is not None else None,
            "interpretation": (
                "A significant positive rho here would support the 'longer "
                "report -> cluster more likely to call it hallucinated' "
                "surface-shortcut hypothesis, independent of ground truth."
            ),
        },
        "ground_truth_tracking_check": rate_analysis,
        "interpretation_gt_tracking": (
            "Compare the GAP between gt_hallucinated_rate and gt_faithful_rate "
            "for medgemma vs. the cluster. A LARGER gap means that model's "
            "'hallucinated' calls track ground truth more; a small gap "
            "(rates similar regardless of GT) is consistent with the cluster "
            "following a fixed bias/shortcut rather than genuine detection."
        ),
        "disagreement_rate_by_hallu_type": disagree_rate_by_type,
    }

    json_path = out_dir / "exp2_medgemma_disagreement.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  → {json_path}")

    csv_path = out_dir / "exp2_medgemma_disagreement_samples.csv"
    df.to_csv(csv_path, index=False)
    print(f"  → {csv_path}")

    # Print the headline comparison directly, since this is the number the
    # paper's Discussion should quote.
    mg_rates = rate_analysis["medgemma_pred_hallucinated_rate"]
    cl_rates = rate_analysis["cluster_majority_pred_hallucinated_rate"]
    if mg_rates.get("gt_hallucinated") is not None and mg_rates.get("gt_faithful") is not None:
        mg_gap = mg_rates["gt_hallucinated"] - mg_rates["gt_faithful"]
        cl_gap = cl_rates["gt_hallucinated"] - cl_rates["gt_faithful"]
        print(f"  medgemma GT-tracking gap:  {mg_gap:+.4f}")
        print(f"  cluster  GT-tracking gap:  {cl_gap:+.4f}")
        print(f"  {'medgemma tracks GT more' if abs(mg_gap) > abs(cl_gap) else 'cluster tracks GT more (unexpected)'}")

    return result


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — Per-source (OpenI vs MIMIC-CXR) generalization check
# ══════════════════════════════════════════════════════════════════════════
#
# The current metrics JSONs all report per_source_accuracy = {"unknown": ...}
# because the 'source' field isn't being propagated from the benchmark file
# into the inference output records. This experiment RE-DERIVES per-source
# accuracy by joining outputs back to the benchmark file on `id` (which DOES
# retain enough structure to recover source, since IDs are prefixed
# "openi_..." / "mimiccxr_..." per build_benchmark_multi.py), so the
# generalization claim can actually be checked without re-running inference.

def exp3_per_source_generalization(root: Path, out_dir: Path,
                                    model_keys=("medgemma", "qwen25_vl", "chexagent",
                                                "llava15", "phi3_vision", "biovil_t",
                                                "biomedclip", "blip2")):
    print("\n" + "=" * 70)
    print("EXP 3: Per-source (OpenI vs MIMIC-CXR) generalization check")
    print("=" * 70)

    bench = load_benchmark(root)
    if bench is None:
        print("  [skip] benchmark file not found under data/v1/processed/ — "
              "cannot recover source labels.")
        return None

    id_to_source = {}
    for _id, rec in bench.items():
        src = rec.get("source")
        if src is None:
            # Fall back to inferring from the id prefix (build_benchmark_multi.py
            # prefixes ids with the source name, e.g. "openi_..." / "mimiccxr_...")
            src = _id.split("_")[0] if "_" in _id else "unknown"
        id_to_source[_id] = src

    source_counts = Counter(id_to_source.values())
    print(f"  Benchmark source distribution: {dict(source_counts)}")
    if len(source_counts) < 2:
        print("  [warn] Only one source detected in the benchmark file — "
              "per-source comparison is not meaningful. Check that the "
              "multi-source build actually ran (build_benchmark_multi.py), "
              "not the legacy single-source build_benchmark.py.")

    all_rows = []
    for mk in model_keys:
        recs = load_outputs_jsonl(root, mk)
        if recs is None:
            print(f"  [skip-model] {mk}_v3.jsonl not found")
            continue
        clean = clean_records(recs)
        if not clean:
            continue

        by_source = defaultdict(list)
        for r in clean:
            src = id_to_source.get(r["id"], "unknown")
            by_source[src].append(r)

        for src, sub_recs in by_source.items():
            y_true, y_pred, y_conf = to_arrays(sub_recs)
            if len(y_true) < 5:
                continue
            acc = float(accuracy_score(y_true, y_pred))
            bal_acc = float(balanced_accuracy_score(y_true, y_pred))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0
            try:
                auroc = float(roc_auc_score(y_true, y_conf)) if len(np.unique(y_true)) > 1 else None
            except ValueError:
                auroc = None
            all_rows.append({
                "model": mk, "source": src, "n": len(sub_recs),
                "accuracy": round(acc, 4), "balanced_acc": round(bal_acc, 4),
                "f1": round(f1, 4), "mcc": round(mcc, 4),
                "auroc": round(auroc, 4) if auroc is not None else None,
            })

    if not all_rows:
        print("  [skip] no model output files found — cannot compute per-source breakdown")
        return None

    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "exp3_per_source_accuracy.csv"
    df.to_csv(csv_path, index=False)
    print(f"  → {csv_path}")

    # Headline: for each model, the GAP between its best and worst source on
    # balanced accuracy. A large gap = generalization failure across sites.
    gaps = []
    for mk, sub in df.groupby("model"):
        if len(sub) < 2:
            continue
        gap = sub["balanced_acc"].max() - sub["balanced_acc"].min()
        gaps.append({
            "model": mk,
            "balanced_acc_gap_across_sources": round(float(gap), 4),
            "per_source": sub.set_index("source")["balanced_acc"].to_dict(),
        })
    gaps_sorted = sorted(gaps, key=lambda x: x["balanced_acc_gap_across_sources"], reverse=True)

    result = {
        "benchmark_source_distribution": dict(source_counts),
        "per_source_metrics": df.to_dict(orient="records"),
        "generalization_gap_ranked": gaps_sorted,
        "interpretation": (
            "A model with a large balanced_acc_gap_across_sources performs "
            "very differently on OpenI vs. MIMIC-CXR studies, indicating its "
            "apparent hallucination-detection ability does not transfer "
            "across imaging sites/institutions — directly relevant to any "
            "clinical deployment claim."
        ),
    }
    json_path = out_dir / "exp3_per_source_accuracy.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  → {json_path}")

    for g in gaps_sorted:
        print(f"  {g['model']:<20} gap={g['balanced_acc_gap_across_sources']:.4f}  {g['per_source']}")

    return result


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4 — Rank-instability driver analysis + Pareto frontier
# ══════════════════════════════════════════════════════════════════════════
#
# summary.csv / halluscore_ablation.csv already show rankings are NOT stable
# across weight schemes. This experiment identifies WHICH component drives
# the instability (via per-component Spearman rank correlation against each
# weight scheme's final ranking) and produces a 3D-projected Pareto frontier
# view (Discrimination, 1-ECE, TypeAccuracy) so the paper can argue for
# reporting a profile instead of a single scalar.

def exp4_rank_instability(root: Path, out_dir: Path):
    print("\n" + "=" * 70)
    print("EXP 4: HalluScore rank-instability driver analysis + Pareto frontier")
    print("=" * 70)

    summary = load_summary_csv(root)
    if summary is None:
        print("  [skip] results/v1/metrics/summary.csv not found")
        return None

    df = summary.copy()
    needed = {"model", "discrimination", "ece", "type_acc_macro", "hallu_score",
              "hs_equal", "hs_calib_heavy", "hs_type_heavy"}
    missing = needed - set(df.columns)
    if missing:
        print(f"  [skip] summary.csv missing expected columns: {missing}")
        return None

    df["one_minus_ece"] = 1.0 - df["ece"]

    # ── Which component's ranking correlates most with the scheme-to-scheme
    #    DISAGREEMENT? i.e. compute rank correlation of each component against
    #    each scheme's score, then compare across schemes -- the component
    #    whose correlation with the final score varies MOST across schemes
    #    is the one "driving" instability (since its weight varies most across
    #    the 5 tested schemes AND it disagrees most with the others).
    schemes = ["hallu_score", "hs_equal", "hs_calib_heavy", "hs_type_heavy"]
    components = {"discrimination": df["discrimination"],
                  "one_minus_ece": df["one_minus_ece"],
                  "type_acc_macro": df["type_acc_macro"]}

    driver_rows = []
    for comp_name, comp_vals in components.items():
        rhos = {}
        for scheme in schemes:
            rho, _ = spearmanr(comp_vals, df[scheme])
            rhos[scheme] = round(float(rho), 4)
        rho_values = list(rhos.values())
        driver_rows.append({
            "component": comp_name,
            **{f"rho_vs_{s}": v for s, v in rhos.items()},
            "rho_range_across_schemes": round(float(max(rho_values) - min(rho_values)), 4),
        })
    driver_df = pd.DataFrame(driver_rows).sort_values(
        "rho_range_across_schemes", ascending=False
    )
    print("  Component correlation with each weight scheme's ranking:")
    print(driver_df.to_string(index=False))

    # ── Pairwise rank swaps between the two most different schemes ──
    scheme_a, scheme_b = "hallu_score", "hs_calib_heavy"  # main vs. calibration-heavy
    rank_a = df.set_index("model")[scheme_a].rank(ascending=False)
    rank_b = df.set_index("model")[scheme_b].rank(ascending=False)
    rank_diff = (rank_a - rank_b).abs().sort_values(ascending=False)
    biggest_movers = rank_diff.head(5).to_dict()

    result = {
        "component_driver_analysis": driver_df.to_dict(orient="records"),
        "most_unstable_component": driver_df.iloc[0]["component"],
        "biggest_rank_movers_main_vs_calib_heavy": {
            k: int(v) for k, v in biggest_movers.items()
        },
        "interpretation": (
            f"'{driver_df.iloc[0]['component']}' shows the widest range of "
            f"rank-correlation with the final HalluScore across the 5 tested "
            f"weight schemes, making it the primary driver of rank "
            f"instability. Models with the largest rank swaps between the "
            f"main (0.5/0.3/0.2) and calibration-heavy (0.3/0.5/0.2) schemes "
            f"are listed above -- these are the models whose leaderboard "
            f"position is most sensitive to a subjective weighting choice, "
            f"which is itself evidence AGAINST reporting a single scalar "
            f"HalluScore as the paper's headline ranking mechanism."
        ),
    }
    json_path = out_dir / "exp4_rank_instability.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  → {json_path}")

    csv_path = out_dir / "exp4_rank_instability.csv"
    driver_df.to_csv(csv_path, index=False)
    print(f"  → {csv_path}")

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(df["discrimination"], df["one_minus_ece"],
                        s=df["type_acc_macro"] * 400 + 30,
                        c=df["hallu_score"], cmap="viridis", edgecolor="black", alpha=0.85)
        for _, r in df.iterrows():
            ax.annotate(r["model"], (r["discrimination"], r["one_minus_ece"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=7.5)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("HalluScore (main scheme)")
        ax.set_xlabel("Discrimination  (↑ better)")
        ax.set_ylabel("1 − ECE  (↑ better)")
        ax.set_title("EXP 4: Pareto view — point size = TypeAccuracy\n"
                      "(no single point dominates on all 3 axes)", fontweight="bold")
        plt.tight_layout()
        fig_path = out_dir / "exp4_pareto_frontier.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {fig_path}")

    return result


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5 — Text-only leakage: trained classifier + paraphrase test
# ══════════════════════════════════════════════════════════════════════════
#
# The existing text_only_baseline is a crude hand-written heuristic
# (n_swap_words * 0.05). This experiment trains an ACTUAL text classifier
# (TF-IDF + logistic regression, held-out test split) on report text alone
# to give a much stronger estimate of how much signal the SWAP_MAP-based
# perturbation leaks lexically, independent of any image.
#
# It also includes a lightweight, self-contained PARAPHRASE-ROBUSTNESS
# proxy: it re-tests the trained classifier after stripping the exact
# injected/original span tokens from the text (simulating what happens if
# hallucinations were phrased without the fixed antonym vocabulary the
# classifier likely keys on). A large accuracy drop under this ablation is
# direct evidence the leakage is specifically tied to the fixed SWAP_MAP
# vocabulary (supporting the paper's proposed fix: paraphrase-preserving
# perturbations), not to some other, unfixable textual property.

def exp5_text_leakage(root: Path, out_dir: Path, test_size: float = 0.25, seed: int = 42):
    print("\n" + "=" * 70)
    print("EXP 5: Text-only leakage — trained classifier + span-ablation test")
    print("=" * 70)

    if not HAS_SKLEARN:
        print("  [skip] scikit-learn not available — install scikit-learn to run this experiment")
        return None

    bench = load_benchmark(root)
    if bench is None:
        print("  [skip] benchmark file not found under data/v1/processed/")
        return None

    records = list(bench.values())
    texts = [r.get("report", "") for r in records]
    labels = [1 if r.get("hallucinated") else 0 for r in records]

    if sum(labels) < 10 or (len(labels) - sum(labels)) < 10:
        print("  [skip] not enough samples in one class to train/test meaningfully")
        return None

    X_train, X_test, y_train, y_test, rec_train, rec_test = train_test_split(
        texts, labels, records, test_size=test_size, random_state=seed, stratify=labels
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, y_train)

    y_pred = clf.predict(Xte)
    y_prob = clf.predict_proba(Xte)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_test, y_pred)) if len(set(y_pred)) > 1 else 0.0
    try:
        auroc = float(roc_auc_score(y_test, y_prob))
    except ValueError:
        auroc = None

    print(f"  Trained TF-IDF+LogReg text-only classifier (held-out test, n={len(y_test)}):")
    print(f"    acc={acc:.4f}  bal_acc={bal_acc:.4f}  f1={f1:.4f}  mcc={mcc:.4f}  "
          f"auroc={auroc if auroc is None else round(auroc,4)}")

    # Top lexical features driving the classifier — should overlap heavily
    # with SWAP_MAP vocabulary if the leakage hypothesis is correct.
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = clf.coef_[0]
    top_pos_idx = np.argsort(coefs)[-20:][::-1]   # pushes toward "hallucinated"
    top_neg_idx = np.argsort(coefs)[:20]           # pushes toward "faithful"
    top_features = {
        "top_tokens_predicting_hallucinated": [
            {"token": feature_names[i], "coef": round(float(coefs[i]), 4)} for i in top_pos_idx
        ],
        "top_tokens_predicting_faithful": [
            {"token": feature_names[i], "coef": round(float(coefs[i]), 4)} for i in top_neg_idx
        ],
    }

    # ── Span-ablation: strip the exact injected/original span tokens from
    #    the TEST text, re-predict with the SAME trained model. If accuracy
    #    collapses toward chance, the model was substantially keying on the
    #    fixed swap vocabulary itself, not on broader linguistic plausibility. ──
    def strip_spans(text: str, rec: dict) -> str:
        t = text
        for key in ("injected_span", "original_span"):
            span = rec.get(key)
            if span:
                t = re.sub(re.escape(span), "", t, flags=re.IGNORECASE)
        return t

    X_test_ablated = [strip_spans(t, r) for t, r in zip(X_test, rec_test)]
    Xte_ablated = vectorizer.transform(X_test_ablated)
    y_pred_ablated = clf.predict(Xte_ablated)
    y_prob_ablated = clf.predict_proba(Xte_ablated)[:, 1]

    acc_ablated = float(accuracy_score(y_test, y_pred_ablated))
    bal_acc_ablated = float(balanced_accuracy_score(y_test, y_pred_ablated))
    try:
        auroc_ablated = float(roc_auc_score(y_test, y_prob_ablated))
    except ValueError:
        auroc_ablated = None

    print(f"  After stripping injected/original span tokens from TEST text only:")
    print(f"    acc={acc_ablated:.4f}  bal_acc={bal_acc_ablated:.4f}  "
          f"auroc={auroc_ablated if auroc_ablated is None else round(auroc_ablated,4)}")

    drop = {
        "acc_drop": round(acc - acc_ablated, 4),
        "bal_acc_drop": round(bal_acc - bal_acc_ablated, 4),
        "auroc_drop": (
            round(auroc - auroc_ablated, 4)
            if auroc is not None and auroc_ablated is not None else None
        ),
    }
    print(f"  Drop from span ablation: {drop}")

    result = {
        "n_train": len(y_train), "n_test": len(y_test),
        "trained_classifier": {
            "accuracy": round(acc, 4), "balanced_accuracy": round(bal_acc, 4),
            "f1": round(f1, 4), "mcc": round(mcc, 4),
            "auroc": round(auroc, 4) if auroc is not None else None,
        },
        "span_ablated_classifier": {
            "accuracy": round(acc_ablated, 4), "balanced_accuracy": round(bal_acc_ablated, 4),
            "auroc": round(auroc_ablated, 4) if auroc_ablated is not None else None,
        },
        "drop_from_span_ablation": drop,
        "top_lexical_features": top_features,
        "interpretation": (
            "A trained (not heuristic) text-only classifier reaching well "
            "above-chance accuracy/AUROC confirms lexical leakage in the "
            "SWAP_MAP-based perturbation strategy. If accuracy collapses "
            "sharply after removing the exact injected/original span tokens "
            "(drop_from_span_ablation large), the leakage is concentrated in "
            "the fixed antonym vocabulary itself and is fixable via "
            "paraphrase-preserving perturbation generation. If the drop is "
            "small, the classifier is exploiting broader linguistic cues "
            "beyond the swap vocabulary, which is a harder problem to fix "
            "and should be reported as a more fundamental benchmark limitation."
        ),
    }
    json_path = out_dir / "exp5_text_leakage.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  → {json_path}")

    return result


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 6 — Paraphrase-repair fix for the Exp 5 leak, re-validated
# ══════════════════════════════════════════════════════════════════════════
#
# Exp 5 found: AUROC 0.918 from text alone (severe leak), dropping to 0.654
# after stripping the exact injected/original span tokens (still above
# chance — residual leak). Inspection of the surviving top features after
# span-ablation showed GRAMMATICAL residue from naive word-substitution:
#   - "livers" (plural "lung" -> singular-context "liver" leaves plural
#     agreement broken: "the livers are clear" instead of "the liver is
#     clear")
#   - "parenchymal effusion" (unnatural collocation: "pleural" -> "parenchymal"
#     substituted into a fixed phrase where the swap doesn't semantically fit)
#   - "chronic cardiopulmonary" (similar collocation artifact)
#
# This experiment builds a SECOND version of the benchmark using a
# grammar-repairing injection function instead of raw regex substitution,
# targeting EXACTLY these two mechanisms:
#   1. Number/article agreement repair after a noun swap (lung->liver singular
#      would leave "a liver", not "livers"; this repairs number agreement
#      using the surrounding determiner/verb as a signal)
#   2. A collocation blocklist: swaps that are known (from Exp 5's own
#      feature list) to produce unnatural collocations are either skipped
#      in favor of the next candidate swap in the same hallucination type,
#      or passed through a light template rewrite that avoids the flagged
#      bigram entirely.
#
# It then RE-RUNS Exp 5's exact classifier + span-ablation pipeline on the
# new benchmark and reports the before/after AUROC, giving a real
# before/after number rather than just a diagnosis.
#
# OPTIONAL LLM PATH: if --exp6_use_llm is passed AND an API key is present
# in the environment (ANTHROPIC_API_KEY), each injected claim is instead
# rewritten by an LLM call that preserves the semantic error (still
# hallucinated relative to the image) while fully naturalizing the phrasing
# -- the strongest possible fix, but it costs real API calls and is
# therefore OFF BY DEFAULT and never invoked silently. Only the outcome
# (before/after AUROC) is reported; raw model outputs are cached to avoid
# re-billing on re-runs.

# Repair rules keyed by the exact injected token (from build_benchmark_multi.py's
# SWAP_MAP), each mapping to a function that takes the full post-injection
# sentence and returns a grammar/collocation-repaired version. Only the
# swaps whose surviving leakage was actually confirmed by Exp 5's feature
# list are repaired here; extending this dict to cover the rest of
# SWAP_MAP is a mechanical follow-up once this validates the approach.
def _fix_number_agreement(text: str, singular_noun: str, plural_wrong: str) -> str:
    """
    Repairs cases like 'the livers are clear' -> 'the liver is clear' when a
    plural-context sentence had its noun swapped to a noun that is normally
    singular-per-patient (liver, spleen, heart, aorta), leaving a stray
    plural verb ('are' instead of 'is') as a giveaway artifact.
    """
    # Fix "livers are" -> "liver is", "livers were" -> "liver was", etc.
    text = re.sub(rf"\b{plural_wrong}\s+are\b", f"{singular_noun} is", text, flags=re.IGNORECASE)
    text = re.sub(rf"\b{plural_wrong}\s+were\b", f"{singular_noun} was", text, flags=re.IGNORECASE)
    text = re.sub(rf"\bthe {plural_wrong}\b", f"the {singular_noun}", text, flags=re.IGNORECASE)
    text = re.sub(rf"\b{plural_wrong}\b", singular_noun, text, flags=re.IGNORECASE)
    return text


# Bigrams confirmed by Exp 5's top-feature list as unnatural collocations
# left behind by a swap. For each, a light rewrite that removes the
# tell-tale phrase while preserving the semantic content of the injected
# hallucination (still wrong relative to the image — that's the point —
# just not wrong in a way that's ALSO grammatically flagged).
_COLLOCATION_REPAIRS = [
    (re.compile(r"\bparenchymal effusion\b", re.IGNORECASE), "parenchymal opacity"),
    (re.compile(r"\bno parenchymal\b", re.IGNORECASE), "no significant parenchymal"),
    (re.compile(r"\bchronic cardiopulmonary\b", re.IGNORECASE), "longstanding cardiopulmonary"),
    (re.compile(r"\bdiaphragml\b", re.IGNORECASE), "diaphragm"),  # confirms a tokenization
    # artifact (likely "diaphragm" + trailing token concatenation from the
    # original swap regex) rather than a linguistic collocation issue --
    # kept here since Exp 5 surfaced it and it should not be left in either
    # version of the benchmark.
]

_NUMBER_AGREEMENT_SWAPS = [
    # (singular_noun, plural_wrong) pairs confirmed leaking via Exp 5
    ("liver", "livers"),
    ("kidney", "kidneys"),
    ("spleen", "spleens"),
]


def repair_injected_text(injected_text: str) -> tuple[str, list[str]]:
    """
    Applies the targeted grammar/collocation repairs above to a single
    already-injected report string. Returns (repaired_text, applied_fixes)
    so we can report exactly how many samples were touched and by which rule
    -- this experiment should be auditable, not a black box.
    """
    applied = []
    text = injected_text

    for singular, plural in _NUMBER_AGREEMENT_SWAPS:
        if re.search(rf"\b{plural}\b", text, re.IGNORECASE):
            new_text = _fix_number_agreement(text, singular, plural)
            if new_text != text:
                applied.append(f"number_agreement:{plural}->{singular}")
                text = new_text

    for pattern, replacement in _COLLOCATION_REPAIRS:
        if pattern.search(text):
            text = pattern.sub(replacement, text)
            applied.append(f"collocation:{pattern.pattern}->{replacement}")

    return text, applied


def _exp6_llm_paraphrase(original_report: str, injected_text: str, injected_span: str,
                          original_span: str, hallu_type: str, client, model_name: str) -> str | None:
    """
    Optional LLM path. Rewrites the injected claim so the SAME semantic
    hallucination is preserved (still factually wrong relative to what the
    original_span said) but phrased without the fixed SWAP_MAP vocabulary
    or any grammatical tell. Returns None on any failure so the caller can
    fall back to the rule-based repair rather than silently using a
    possibly-malformed LLM output.
    """
    prompt = (
        "You are helping construct a hallucination-detection benchmark for "
        "radiology reports. Below is a chest X-ray report that has had ONE "
        "factual error deliberately injected into it for evaluation purposes.\n\n"
        f"Original correct claim: \"{original_span}\"\n"
        f"Injected (incorrect) claim: \"{injected_span}\"\n"
        f"Full report with the injected error: \"{injected_text}\"\n"
        f"Error category: {hallu_type}\n\n"
        "Rewrite the FULL report so that the SAME factual error is present "
        "(the report should still describe something that contradicts what "
        "a real image showing the original claim would show), but phrase it "
        "naturally -- do not just swap one word for another; make sure "
        "grammar, number agreement, and word collocations all read like a "
        "real radiology report. Do not fix or remove the error itself, only "
        "make its phrasing natural. Output ONLY the rewritten report text, "
        "nothing else."
    )
    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        out = "\n".join(text_blocks).strip()
        return out if out else None
    except Exception as e:
        print(f"    [llm-error] {e}")
        return None


def exp6_paraphrase_fix(root: Path, out_dir: Path, use_llm: bool = False,
                          llm_model: str = "claude-sonnet-4-6", llm_max_samples: int = None,
                          test_size: float = 0.25, seed: int = 42):
    print("\n" + "=" * 70)
    print("EXP 6: Paraphrase-repair fix for text leakage — re-validated")
    print("=" * 70)

    if not HAS_SKLEARN:
        print("  [skip] scikit-learn not available")
        return None

    bench = load_benchmark(root)
    if bench is None:
        print("  [skip] benchmark file not found under data/v1/processed/")
        return None

    records = list(bench.values())

    llm_client = None
    if use_llm:
        try:
            import os
            import anthropic
            if not os.environ.get("ANTHROPIC_API_KEY"):
                print("  [warn] --exp6_use_llm passed but ANTHROPIC_API_KEY is not "
                      "set — falling back to the rule-based repair path instead "
                      "of silently skipping or erroring.")
                use_llm = False
            else:
                llm_client = anthropic.Anthropic()
                print(f"  [llm] Using {llm_model} to paraphrase injected claims "
                      f"(this will make real API calls and incur cost).")
        except ImportError:
            print("  [warn] --exp6_use_llm passed but the 'anthropic' package is "
                  "not installed (`pip install anthropic --break-system-packages`) "
                  "— falling back to the rule-based repair path.")
            use_llm = False

    repaired_records = []
    n_rule_repaired = 0
    n_llm_repaired = 0
    n_llm_failed = 0
    rule_repair_log = Counter()

    hallu_records = [r for r in records if r.get("hallucinated")]
    llm_budget = llm_max_samples if llm_max_samples is not None else len(hallu_records)

    for i, r in enumerate(records):
        new_r = dict(r)
        if r.get("hallucinated") and r.get("report"):
            if use_llm and llm_client is not None and i < llm_budget:
                llm_out = _exp6_llm_paraphrase(
                    r.get("original_report", ""), r["report"], r.get("injected_span", ""),
                    r.get("original_span", ""), r.get("hallu_type", ""),
                    llm_client, llm_model,
                )
                if llm_out:
                    new_r["report"] = llm_out
                    new_r["repair_method"] = "llm"
                    n_llm_repaired += 1
                else:
                    n_llm_failed += 1
                    repaired, fixes = repair_injected_text(r["report"])
                    new_r["report"] = repaired
                    new_r["repair_method"] = "rule_fallback_after_llm_fail"
                    if fixes:
                        n_rule_repaired += 1
                        rule_repair_log.update(fixes)
            else:
                repaired, fixes = repair_injected_text(r["report"])
                new_r["report"] = repaired
                new_r["repair_method"] = "rule"
                if fixes:
                    n_rule_repaired += 1
                    rule_repair_log.update(fixes)
        repaired_records.append(new_r)

    print(f"  Repaired {n_rule_repaired} samples via rule-based grammar/collocation fixes")
    if use_llm:
        print(f"  Repaired {n_llm_repaired} samples via LLM paraphrase "
              f"({n_llm_failed} LLM calls failed and fell back to rule-based repair)")
    print(f"  Rule application counts: {dict(rule_repair_log)}")

    repaired_path = out_dir / "exp6_paraphrase_fix_benchmark.jsonl"
    with open(repaired_path, "w") as f:
        for r in repaired_records:
            f.write(json.dumps(r) + "\n")
    print(f"  → {repaired_path}")

    # ── Re-run Exp 5's EXACT classifier + span-ablation pipeline on the
    #    repaired benchmark, so the before/after comparison isolates the
    #    effect of the repair, not a different train/test split or model. ──
    texts = [r.get("report", "") for r in repaired_records]
    labels = [1 if r.get("hallucinated") else 0 for r in repaired_records]

    if sum(labels) < 10 or (len(labels) - sum(labels)) < 10:
        print("  [skip] not enough samples in one class after repair")
        return None

    X_train, X_test, y_train, y_test, rec_train, rec_test = train_test_split(
        texts, labels, repaired_records, test_size=test_size, random_state=seed, stratify=labels
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, y_train)

    y_pred = clf.predict(Xte)
    y_prob = clf.predict_proba(Xte)[:, 1]

    acc_after = float(accuracy_score(y_test, y_pred))
    bal_acc_after = float(balanced_accuracy_score(y_test, y_pred))
    f1_after = float(f1_score(y_test, y_pred, zero_division=0))
    try:
        auroc_after = float(roc_auc_score(y_test, y_prob))
    except ValueError:
        auroc_after = None

    print(f"  AFTER repair — trained text-only classifier (held-out test, n={len(y_test)}):")
    print(f"    acc={acc_after:.4f}  bal_acc={bal_acc_after:.4f}  f1={f1_after:.4f}  "
          f"auroc={auroc_after if auroc_after is None else round(auroc_after, 4)}")

    # Load Exp 5's BEFORE numbers if available, for a direct side-by-side
    exp5_path = out_dir / "exp5_text_leakage.json"
    before = None
    if exp5_path.exists():
        with open(exp5_path) as f:
            exp5_data = json.load(f)
        before = exp5_data.get("trained_classifier")

    comparison = {
        "before_repair": before,
        "after_repair": {
            "accuracy": round(acc_after, 4), "balanced_accuracy": round(bal_acc_after, 4),
            "f1": round(f1_after, 4),
            "auroc": round(auroc_after, 4) if auroc_after is not None else None,
        },
        "auroc_reduction": (
            round(before["auroc"] - auroc_after, 4)
            if before and before.get("auroc") is not None and auroc_after is not None else None
        ),
        "n_rule_repaired": n_rule_repaired,
        "n_llm_repaired": n_llm_repaired,
        "n_llm_failed": n_llm_failed,
        "rule_application_counts": dict(rule_repair_log),
        "repair_mode": "llm" if use_llm else "rule_based",
        "interpretation": (
            "Compares the Exp 5 text-only classifier's AUROC before vs. after "
            "repairing the specific grammatical/collocation artifacts Exp 5's "
            "own feature analysis surfaced. A meaningful auroc_reduction here "
            "is direct evidence the leak was fixable at the injection-"
            "generation stage, not an inherent property of the task. If AUROC "
            "after repair is still well above chance (>0.6), a further, "
            "stronger fix (full LLM paraphrase of injected claims, not just "
            "grammar repair -- see --exp6_use_llm) is warranted before this "
            "benchmark's cross-model comparisons can be trusted at face value."
        ),
    }
    json_path = out_dir / "exp6_paraphrase_fix_validation.json"
    with open(json_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"  → {json_path}")

    if before and before.get("auroc") is not None and auroc_after is not None:
        print(f"  AUROC: {before['auroc']:.4f} (before repair) -> "
              f"{auroc_after:.4f} (after repair)   Δ = {before['auroc'] - auroc_after:+.4f}")
    else:
        print("  [note] Exp 5 must be run first (same output dir) to get a "
              "before/after comparison; ran Exp 6 standalone this time, so "
              "only the 'after' numbers are available.")

    return comparison


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 7 — Validate the MCC-gated Discrimination fix on REAL models
# ══════════════════════════════════════════════════════════════════════════
#
# Exp 1's synthetic sweep proposed gating Discrimination on an MCC floor
# (Discrimination = 0 whenever MCC < mcc_floor, else the original formula)
# as a candidate fix for the blind spot. This experiment applies that gate
# directly to every REAL model's already-computed metrics (from the
# *_metrics.json files), recomputes HalluScore under the gated formula, and
# reports the resulting re-ranking -- closing the loop between "we proposed
# a fix on synthetic data" and "here is what it actually does to the
# leaderboard."
#
# Sweeps mcc_floor over a small grid (0.02, 0.05, 0.10, 0.15) since the
# right floor is itself a judgment call the paper should show is not
# hyper-sensitive, rather than picking one value and hoping it's defensible.

def exp7_validate_mcc_gate(root: Path, out_dir: Path,
                            mcc_floors=(0.02, 0.05, 0.10, 0.15),
                            weights=(0.5, 0.3, 0.2)):
    print("\n" + "=" * 70)
    print("EXP 7: Validating the MCC-gated Discrimination fix on real models")
    print("=" * 70)

    model_keys = ["qwen25_vl", "medgemma", "llava15", "phi3_vision", "chexagent",
                  "biovil_t", "biomedclip", "blip2", "constant_faithful", "random",
                  "text_only_baseline", "shuffled_image_baseline_qwen25_vl",
                  "shuffled_image_baseline_medgemma"]

    rows = []
    for mk in model_keys:
        m = load_metrics_json(root, mk)
        if m is None:
            continue
        mcc = m["mcc"]
        f1 = m["f1"]
        ece = m["ece"]
        type_acc = m["type_accuracy"]["macro_avg"]
        disc_original = m["discrimination"]

        w_disc, w_ece, w_type = weights
        hs_original = round(w_disc * disc_original + w_ece * (1 - ece) + w_type * type_acc, 4)

        row = {
            "model": mk,
            "model_category": m.get("model_category", "unknown"),
            "mcc": mcc, "f1": f1, "ece": ece, "type_acc_macro": type_acc,
            "discrimination_original": disc_original,
            "hallu_score_original": hs_original,
        }
        for floor in mcc_floors:
            disc_gated = 0.0 if mcc < floor else disc_original
            hs_gated = round(w_disc * disc_gated + w_ece * (1 - ece) + w_type * type_acc, 4)
            row[f"discrimination_gated_floor{floor}"] = round(disc_gated, 4)
            row[f"hallu_score_gated_floor{floor}"] = hs_gated
        rows.append(row)

    if not rows:
        print("  [skip] no model metrics JSON files found")
        return None

    df = pd.DataFrame(rows)
    csv_path = out_dir / "exp7_mcc_gate_validation.csv"
    df.to_csv(csv_path, index=False)
    print(f"  → {csv_path}")

    # ── Headline check: for each floor, does the best VLM by the GATED score
    #    still make sense (i.e. is it NOT a near-degenerate classifier), and
    #    does MedGemma's rank improve relative to CheXagent/near-degenerate
    #    models specifically? ──
    real_vlm_mask = df["model_category"] == "vlm"
    results_per_floor = {}
    for floor in mcc_floors:
        col = f"hallu_score_gated_floor{floor}"
        ranked = df[real_vlm_mask].sort_values(col, ascending=False)
        top_model = ranked.iloc[0]["model"]
        top_score = ranked.iloc[0][col]
        medgemma_rank = int((ranked["model"] == "medgemma").idxmax()) if "medgemma" in ranked["model"].values else None
        medgemma_rank = list(ranked["model"]).index("medgemma") + 1 if "medgemma" in list(ranked["model"]) else None
        chexagent_rank = list(ranked["model"]).index("chexagent") + 1 if "chexagent" in list(ranked["model"]) else None
        chexagent_mcc = df.loc[df["model"] == "chexagent", "mcc"].values
        chexagent_gated_disc = df.loc[df["model"] == "chexagent", f"discrimination_gated_floor{floor}"].values
        results_per_floor[floor] = {
            "top_vlm_by_gated_score": top_model,
            "top_vlm_gated_score": float(top_score),
            "medgemma_rank_among_vlms": medgemma_rank,
            "chexagent_rank_among_vlms": chexagent_rank,
            "chexagent_mcc": float(chexagent_mcc[0]) if len(chexagent_mcc) else None,
            "chexagent_discrimination_after_gate": (
                float(chexagent_gated_disc[0]) if len(chexagent_gated_disc) else None
            ),
            "chexagent_gated_below_floor": (
                bool(chexagent_mcc[0] < floor) if len(chexagent_mcc) else None
            ),
        }
        print(f"  floor={floor}: top VLM = {top_model} ({top_score:.4f})  "
              f"medgemma_rank={medgemma_rank}  chexagent_rank={chexagent_rank}  "
              f"chexagent_gated_out={results_per_floor[floor]['chexagent_gated_below_floor']}")

    # ── Rank stability check for medgemma specifically across floors, since
    #    the whole point is "does the fix let the genuinely-best model win
    #    without being hyper-sensitive to the exact floor chosen".
    #
    #    IMPORTANT: a floor below chexagent's actual MCC (0.033) does not
    #    activate the gate at all -- including such a floor in a blanket
    #    "stable across ALL tested floors" check conflates two different
    #    questions: (a) is there a floor low enough to be a no-op? (yes,
    #    trivially, by construction) vs (b) once the gate is high enough to
    #    ACTUALLY gate out the near-degenerate model, is the result stable
    #    to the exact value chosen? Only (b) is the question the paper
    #    should be answering. We therefore report stability separately over
    #    just the floors where chexagent (the known near-degenerate case)
    #    was actually gated out, plus the minimum floor that activates the
    #    fix at all -- rather than one blanket boolean that silently mixes
    #    "inactive" and "active" floors together. ──
    medgemma_ranks = [results_per_floor[f]["medgemma_rank_among_vlms"] for f in mcc_floors]
    active_floors = [f for f in mcc_floors if results_per_floor[f]["chexagent_gated_below_floor"]]
    inactive_floors = [f for f in mcc_floors if not results_per_floor[f]["chexagent_gated_below_floor"]]
    medgemma_ranks_active_only = [results_per_floor[f]["medgemma_rank_among_vlms"] for f in active_floors]
    medgemma_rank_stable_among_active_floors = (
        len(set(medgemma_ranks_active_only)) == 1 if medgemma_ranks_active_only else None
    )
    min_floor_that_activates_fix = min(active_floors) if active_floors else None

    result = {
        "weights_used": {"discrimination": weights[0], "ece": weights[1], "type_accuracy": weights[2]},
        "mcc_floors_tested": list(mcc_floors),
        "per_floor_results": results_per_floor,
        "medgemma_rank_across_floors": medgemma_ranks,
        "inactive_floors_too_low_to_gate_chexagent": inactive_floors,
        "active_floors_that_gate_chexagent_out": active_floors,
        "min_floor_that_activates_fix": min_floor_that_activates_fix,
        "medgemma_rank_stable_among_active_floors": medgemma_rank_stable_among_active_floors,
        "original_top_vlm_by_halluscore": (
            df[real_vlm_mask].sort_values("hallu_score_original", ascending=False).iloc[0]["model"]
        ),
        "interpretation": (
            "Under the ORIGINAL (ungated) formula, chexagent topped the "
            "real-VLM leaderboard despite MCC≈0.03 (near-zero signal) -- "
            "this is the exact blind spot Exp 1's synthetic sweep predicted. "
            "'chexagent_gated_below_floor'=True at a given floor means that "
            "floor successfully zeroes out chexagent's Discrimination term, "
            "removing it from contention for top VLM. Floors BELOW "
            "chexagent's actual MCC (~0.033) are trivially inactive (listed "
            "in inactive_floors_too_low_to_gate_chexagent) and are NOT part "
            "of the stability claim -- a too-low floor being a no-op is "
            "expected, not evidence against the fix. The real question is "
            "whether medgemma's rank is stable ACROSS THE FLOORS THAT ARE "
            "HIGH ENOUGH TO ACTUALLY GATE chexagent out "
            "(medgemma_rank_stable_among_active_floors); if True, the fix "
            "works and is not sensitive to the exact value chosen once it "
            "clears the minimum threshold needed to matter "
            "(min_floor_that_activates_fix)."
        ),
    }
    json_path = out_dir / "exp7_mcc_gate_validation.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  → {json_path}")
    print(f"  min floor that activates the fix: {min_floor_that_activates_fix}")
    print(f"  medgemma rank stable among ACTIVE (gate-triggering) floors: "
          f"{medgemma_rank_stable_among_active_floors}  "
          f"(inactive/no-op floors excluded: {inactive_floors})")

    return result


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 8 — Closed-vocabulary hypothesis test (no API needed)
# ══════════════════════════════════════════════════════════════════════════
#
# Exp 6 found that repairing grammar/collocation artifacts barely moved the
# leak (AUROC 0.9183 -> 0.9162, Δ=0.002) -- ruling out "grammatical residue"
# as the primary leakage mechanism. The competing hypothesis, motivated by
# Exp 5's own top-feature list ("abnormal", "severe", "normal", "enlarged"
# dominating over any single swap-specific token), is that the leak is
# simply the FIXED, CLOSED antonym vocabulary itself: because SWAP_MAP has
# a small, fixed set of substitution words shared across ALL samples of a
# given hallucination type, ANY bag-of-words classifier will learn "these
# ~40 words correlate with the label" regardless of grammar.
#
# This experiment tests that hypothesis directly and for free (no LLM calls
# needed) by blocking the ENTIRE SWAP_MAP vocabulary (all keys and values,
# across all three hallucination types) from the TF-IDF vectorizer's
# feature space -- not just the specific span injected into a given sample
# (which is what Exp 5's span-ablation already tried and only got to 0.654),
# but literally every word that EVER appears as a swap source or target
# anywhere in the benchmark, even in samples where a DIFFERENT word was
# swapped. This is a strictly harder ablation than Exp 5's per-sample span
# removal, and isolates the vocabulary-closure hypothesis cleanly:
#
#   - If AUROC collapses to near-chance (~0.5) once the closed vocabulary is
#     blocked entirely, the leak IS the fixed vocabulary -- confirming
#     paraphrase/open-vocabulary generation (LLM-based) is necessary and
#     sufficient, and no further grammar engineering will help.
#   - If AUROC remains well above chance even with the full SWAP_MAP
#     vocabulary blocked, something else entirely is leaking (e.g. report
#     LENGTH, punctuation patterns, or a stylistic tell from the injection
#     process itself) and needs a different diagnosis before any fix will
#     work -- also an important, reportable finding either way.
#
# Also runs a control: the SAME classifier/pipeline with a RANDOM
# vocabulary block of equal size (same number of words, chosen at random
# from the corpus) -- this confirms any AUROC drop is specifically caused by
# blocking the swap vocabulary and not just an artifact of shrinking the
# feature space by however many words SWAP_MAP happens to contain.

def exp8_closed_vocabulary_test(root: Path, out_dir: Path, test_size: float = 0.25, seed: int = 42):
    print("\n" + "=" * 70)
    print("EXP 8: Closed-vocabulary hypothesis test (no API needed)")
    print("=" * 70)

    if not HAS_SKLEARN:
        print("  [skip] scikit-learn not available")
        return None

    bench = load_benchmark(root)
    if bench is None:
        print("  [skip] benchmark file not found under data/v1/processed/")
        return None

    # Pull the SWAP_MAP vocabulary the same way run_inference.py's
    # run_text_only_baseline() does, so this experiment stays consistent
    # with the rest of the pipeline's definition of "swap vocabulary".
    swap_words = set()
    try:
        import sys as _sys
        code_root_candidates = [
            root / "code/v1/01_data_prep",
            root / "01_data_prep",
        ]
        for p in code_root_candidates:
            if p.exists():
                _sys.path.insert(0, str(p))
        from importlib import import_module
        try:
            bb_mod = import_module("build_benchmark_multi")
        except ImportError:
            bb_mod = import_module("build_benchmark")
        for d in bb_mod.SWAP_MAP.values():
            swap_words.update(w.lower() for w in d.keys())
            swap_words.update(w.lower() for w in d.values())
        print(f"  Loaded SWAP_MAP vocabulary: {len(swap_words)} unique words "
              f"across all hallucination types")
        print(f"    {sorted(swap_words)}")
    except Exception as e:
        print(f"  [skip] could not import build_benchmark_multi.py / "
              f"build_benchmark.py to recover SWAP_MAP: {e}")
        return None

    if not swap_words:
        print("  [skip] SWAP_MAP vocabulary came back empty — check the import path")
        return None

    records = list(bench.values())
    texts = [r.get("report", "") for r in records]
    labels = [1 if r.get("hallucinated") else 0 for r in records]

    if sum(labels) < 10 or (len(labels) - sum(labels)) < 10:
        print("  [skip] not enough samples in one class")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    def run_classifier(vocab_exclude: set, condition_name: str) -> dict:
        """Trains TF-IDF+LogReg excluding the given vocabulary from the
        feature space entirely (not just stripping it from the text --
        excluding it from the VECTORIZER means even partial/compound
        n-grams containing it are handled correctly, and the exclusion is
        enforced at the SAME strength for every sample, unlike a regex
        strip which can miscount inflected forms)."""
        vectorizer = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2), min_df=2,
            stop_words=list(vocab_exclude) if vocab_exclude else None,
        )
        Xtr = vectorizer.fit_transform(X_train)
        Xte = vectorizer.transform(X_test)

        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(Xtr, y_train)
        y_pred = clf.predict(Xte)
        y_prob = clf.predict_proba(Xte)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        bal_acc = float(balanced_accuracy_score(y_test, y_pred))
        try:
            auroc = float(roc_auc_score(y_test, y_prob))
        except ValueError:
            auroc = None

        n_features_actually_used = len(vectorizer.get_feature_names_out())
        print(f"  [{condition_name}] n_features={n_features_actually_used}  "
              f"acc={acc:.4f}  bal_acc={bal_acc:.4f}  "
              f"auroc={auroc if auroc is None else round(auroc, 4)}")

        return {
            "condition": condition_name,
            "n_vocab_excluded": len(vocab_exclude),
            "n_features_used": n_features_actually_used,
            "accuracy": round(acc, 4), "balanced_accuracy": round(bal_acc, 4),
            "auroc": round(auroc, 4) if auroc is not None else None,
        }

    print("\n  Condition 1/3: baseline (no vocabulary excluded)")
    baseline = run_classifier(set(), "baseline_no_exclusion")

    print("\n  Condition 2/3: SWAP_MAP vocabulary excluded entirely (the real test)")
    swap_excluded = run_classifier(swap_words, "swap_vocabulary_excluded")

    print(f"\n  Condition 3/3: random control — {len(swap_words)} random words excluded "
          f"(same size as SWAP_MAP, to isolate 'blocking N words' from "
          f"'blocking THESE SPECIFIC N words')")
    rng = np.random.default_rng(seed)
    # Build the random control vocabulary from the actual corpus so it's a
    # fair comparison (blocking N words that DO appear vs N that don't would
    # trivially show no effect and wouldn't be a real control).
    probe_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 1), min_df=2)
    probe_vectorizer.fit(X_train)
    corpus_vocab = list(probe_vectorizer.get_feature_names_out())
    non_swap_corpus_vocab = [w for w in corpus_vocab if w not in swap_words]
    random_control_words = set(
        rng.choice(non_swap_corpus_vocab, size=min(len(swap_words), len(non_swap_corpus_vocab)), replace=False)
    )
    random_control = run_classifier(random_control_words, "random_control_same_size")

    swap_drop = (
        round(baseline["auroc"] - swap_excluded["auroc"], 4)
        if baseline["auroc"] is not None and swap_excluded["auroc"] is not None else None
    )
    random_drop = (
        round(baseline["auroc"] - random_control["auroc"], 4)
        if baseline["auroc"] is not None and random_control["auroc"] is not None else None
    )

    # Chance-level reference: with a roughly balanced class split, AUROC=0.5
    # is chance. We call the leak "explained by closed vocabulary" if
    # swap_excluded's AUROC is close to 0.5 AND the random control's AUROC
    # stays much closer to baseline (confirming the drop is specific to the
    # SWAP_MAP words, not just "removing any ~40 words hurts a little").
    near_chance_threshold = 0.60  # generous; still well below baseline's ~0.92
    swap_vocab_explains_leak = (
        swap_excluded["auroc"] is not None and swap_excluded["auroc"] < near_chance_threshold
        and (random_drop is None or (swap_drop is not None and swap_drop > 3 * abs(random_drop) if random_drop else True))
    )

    result = {
        "swap_map_vocabulary_size": len(swap_words),
        "swap_map_vocabulary": sorted(swap_words),
        "baseline": baseline,
        "swap_vocabulary_excluded": swap_excluded,
        "random_control_same_size": random_control,
        "auroc_drop_from_excluding_swap_vocab": swap_drop,
        "auroc_drop_from_random_control": random_drop,
        "near_chance_threshold_used": near_chance_threshold,
        "swap_vocabulary_explains_the_leak": swap_vocab_explains_leak,
        "interpretation": (
            (
                "AUROC collapses to near-chance once the FULL closed SWAP_MAP "
                "vocabulary is excluded, while the size-matched random control "
                "barely moves from baseline. This confirms the leak Exp 5 found "
                "is caused by the closed, fixed antonym vocabulary itself, not "
                "by grammar/collocation artifacts (which Exp 6 already ruled "
                "out) and not merely by feature-space shrinkage (ruled out by "
                "the random control). Only an open-vocabulary / paraphrase-"
                "based injection strategy (e.g. the --exp6_use_llm path) can "
                "fix this; further grammar engineering will not help."
            ) if swap_vocab_explains_leak else (
                "Excluding the full SWAP_MAP vocabulary did NOT collapse AUROC "
                "to near-chance (or the random control dropped by a comparable "
                "amount), meaning the leak is NOT fully explained by the fixed "
                "swap vocabulary alone. Some other signal -- report length, "
                "punctuation/formatting differences introduced by the injection "
                "process, or co-occurring words not in the swap list -- is also "
                "contributing meaningfully. This should be investigated before "
                "assuming an LLM-paraphrase fix alone will resolve the leak."
            )
        ),
    }
    json_path = out_dir / "exp8_closed_vocabulary_test.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  → {json_path}")
    print(f"  AUROC: baseline={baseline['auroc']}  "
          f"swap_excluded={swap_excluded['auroc']} (drop={swap_drop})  "
          f"random_control={random_control['auroc']} (drop={random_drop})")
    print(f"  swap_vocabulary_explains_the_leak: {swap_vocab_explains_leak}")

    return result


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 9 — Eligibility-confound test for the residual leak (no API needed)
# ══════════════════════════════════════════════════════════════════════════
#
# Exp 8 found: excluding the ENTIRE SWAP_MAP vocabulary drops AUROC from
# 0.918 to 0.689 -- a real, substantial reduction (Δ=0.229), but AUROC 0.689
# is still far above chance. Exp 6 already ruled out grammar/collocation
# residue as the explanation for a much smaller Δ (0.002). So roughly half
# of the total leak is explained by neither hypothesis. This experiment
# tests the next most likely mechanism: an ELIGIBILITY CONFOUND in how
# build_benchmark_multi.py's inject() function selects WHICH reports get
# perturbed at all, independent of what word ends up being swapped.
#
# inject() only succeeds on a report if that report contains at least one
# word from SWAP_MAP for the chosen hallu_type -- e.g. a report can only
# become an "object" hallucination if it contains a word like "lung",
# "heart", "pleura", etc. This means the hallucinated-class reports are NOT
# a random sample of all reports -- they are systematically drawn from the
# subset of reports that happen to CONTAIN invertible vocabulary in the
# first place, which may correlate with report length, structural
# complexity, or the presence of OTHER (non-swapped) anatomical/descriptive
# language that differs from the faithful-class distribution for reasons
# that have nothing to do with the injected error itself.
#
# This experiment tests three specific, falsifiable sub-hypotheses, ALL
# computed on text with the full SWAP_MAP vocabulary already stripped (so
# any signal found here is PROVABLY not lexical -- it isolates exactly the
# kind of structural/selection confound Exp 8's negative result pointed to):
#
#   9a. Report length (word count) differs by class, even after stripping
#       SWAP_MAP words. A length-only classifier (one feature: word count)
#       tests this directly and gives a clean, interpretable AUROC.
#   9b. Count of OTHER anatomical/descriptive terms NOT in SWAP_MAP (a
#       broader medical-vocabulary list) differs by class -- i.e. eligible-
#       for-injection reports are simply "more anatomically detailed"
#       reports in general, which the classifier could exploit via n-grams
#       that have nothing to do with SWAP_MAP.
#   9c. Retrain the FULL TF-IDF+LogReg classifier (as in Exp 8) but this
#       time ALSO exclude every token from a broad radiology-vocabulary
#       list (not just SWAP_MAP) to see how much further AUROC drops --
#       if it approaches chance now, the residual WAS a vocabulary-breadth
#       confound (just a larger vocabulary than SWAP_MAP alone, not a
#       structural/length confound). If it does NOT approach chance even
#       then, length/structure (9a) is the more likely explanation and
#       should be reported as a distinct, harder-to-fix benchmark property.

# Broad radiology-report vocabulary beyond SWAP_MAP -- covers common
# anatomical/descriptive terms that could co-occur with SWAP_MAP-eligible
# reports without being SWAP_MAP words themselves. NOT exhaustive by
# design (an exhaustive medical NER pass is a heavier follow-up); this is
# a deliberately broad, cheap probe list to test the "more anatomical
# vocabulary in general" hypothesis before committing to something heavier.
_BROADER_RADIOLOGY_VOCAB = {
    "cardiomegaly", "opacity", "opacities", "infiltrate", "infiltrates",
    "silhouette", "costophrenic", "angle", "angles", "vascular", "markings",
    "hyperinflation", "hyperinflated", "granuloma", "granulomas", "scarring",
    "fibrosis", "calcification", "calcified", "degenerative", "osseous",
    "cardiac", "thoracic", "pulmonary", "bronchovascular", "hemidiaphragm",
    "costochondral", "sulcus", "apex", "apices", "base", "bases", "hilum",
    "hila", "vasculature", "contour", "contours", "border", "borders",
    "prominent", "prominence", "engorged", "engorgement", "atherosclerotic",
    "atherosclerosis", "degeneration", "spondylosis", "kyphosis", "scoliosis",
}


def exp9_eligibility_confound(root: Path, out_dir: Path, test_size: float = 0.25, seed: int = 42):
    print("\n" + "=" * 70)
    print("EXP 9: Eligibility-confound test for the residual leak (no API needed)")
    print("=" * 70)

    if not HAS_SKLEARN:
        print("  [skip] scikit-learn not available")
        return None

    bench = load_benchmark(root)
    if bench is None:
        print("  [skip] benchmark file not found under data/v1/processed/")
        return None

    # Recover SWAP_MAP the same way Exp 8 does, for consistency.
    swap_words = set()
    try:
        import sys as _sys
        for p in (root / "code/v1/01_data_prep", root / "01_data_prep"):
            if p.exists():
                _sys.path.insert(0, str(p))
        from importlib import import_module
        try:
            bb_mod = import_module("build_benchmark_multi")
        except ImportError:
            bb_mod = import_module("build_benchmark")
        for d in bb_mod.SWAP_MAP.values():
            swap_words.update(w.lower() for w in d.keys())
            swap_words.update(w.lower() for w in d.values())
    except Exception as e:
        print(f"  [skip] could not recover SWAP_MAP: {e}")
        return None

    records = list(bench.values())
    labels_all = [1 if r.get("hallucinated") else 0 for r in records]
    if sum(labels_all) < 10 or (len(labels_all) - sum(labels_all)) < 10:
        print("  [skip] not enough samples in one class")
        return None

    def strip_words(text: str, vocab: set) -> str:
        if not vocab:
            return text
        pattern = re.compile(r"\b(" + "|".join(re.escape(w) for w in vocab) + r")\b", re.IGNORECASE)
        return pattern.sub("", text)

    # Text with ONLY SWAP_MAP stripped (matches Exp 8's "swap_vocabulary_excluded"
    # condition exactly, so 9a/9b are measured on the SAME text Exp 8 found
    # AUROC=0.689 on -- this experiment explains THAT number, not a new one).
    stripped_texts = [strip_words(r.get("report", ""), swap_words) for r in records]

    # ── 9a: report length (word count) after stripping SWAP_MAP words ──
    print("\n  9a: Report length (word count) by class, SWAP_MAP words already stripped")
    word_counts = np.array([len(t.split()) for t in stripped_texts])
    y = np.array(labels_all)
    len_hallu = word_counts[y == 1]
    len_faith = word_counts[y == 0]
    print(f"    mean word count — hallucinated: {len_hallu.mean():.2f}  "
          f"faithful: {len_faith.mean():.2f}")

    # Single-feature classifier: word count only. Uses the SAME train/test
    # split logic as Exp 8 for a fair, direct comparison.
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        np.arange(len(records)), y, test_size=test_size, random_state=seed, stratify=y
    )
    wc_train = word_counts[X_train_idx].reshape(-1, 1)
    wc_test = word_counts[X_test_idx].reshape(-1, 1)
    len_clf = LogisticRegression(class_weight="balanced")
    len_clf.fit(wc_train, y_train)
    len_pred_prob = len_clf.predict_proba(wc_test)[:, 1]
    try:
        len_auroc = float(roc_auc_score(y_test, len_pred_prob))
    except ValueError:
        len_auroc = None
    print(f"    length-only classifier AUROC: {len_auroc}")

    # ── 9b: count of broader (non-SWAP_MAP) radiology vocabulary by class ──
    print("\n  9b: Broader (non-SWAP_MAP) radiology-vocabulary term count by class")

    def count_broader_vocab(text: str) -> int:
        tokens = re.findall(r"[a-z]+", text.lower())
        return sum(1 for t in tokens if t in _BROADER_RADIOLOGY_VOCAB)

    broader_counts = np.array([count_broader_vocab(t) for t in stripped_texts])
    bc_hallu = broader_counts[y == 1]
    bc_faith = broader_counts[y == 0]
    print(f"    mean broader-vocab count — hallucinated: {bc_hallu.mean():.3f}  "
          f"faithful: {bc_faith.mean():.3f}")

    bc_train = broader_counts[X_train_idx].reshape(-1, 1)
    bc_test = broader_counts[X_test_idx].reshape(-1, 1)
    bc_clf = LogisticRegression(class_weight="balanced")
    bc_clf.fit(bc_train, y_train)
    bc_pred_prob = bc_clf.predict_proba(bc_test)[:, 1]
    try:
        bc_auroc = float(roc_auc_score(y_test, bc_pred_prob))
    except ValueError:
        bc_auroc = None
    print(f"    broader-vocab-count-only classifier AUROC: {bc_auroc}")

    # Combined length + broader-vocab-count (2-feature) classifier, to see
    # how much of Exp 8's 0.689 residual these two simple structural
    # features alone can explain.
    combo_train = np.column_stack([wc_train.ravel(), bc_train.ravel()])
    combo_test = np.column_stack([wc_test.ravel(), bc_test.ravel()])
    combo_clf = LogisticRegression(class_weight="balanced")
    combo_clf.fit(combo_train, y_train)
    combo_pred_prob = combo_clf.predict_proba(combo_test)[:, 1]
    try:
        combo_auroc = float(roc_auc_score(y_test, combo_pred_prob))
    except ValueError:
        combo_auroc = None
    print(f"    combined (length + broader-vocab-count) classifier AUROC: {combo_auroc}")

    # ── 9c: full TF-IDF classifier with SWAP_MAP + broader vocab BOTH excluded ──
    print("\n  9c: Full TF-IDF classifier — SWAP_MAP + broader radiology vocab both excluded")
    all_texts = [r.get("report", "") for r in records]
    X_train_txt = [all_texts[i] for i in X_train_idx]
    X_test_txt = [all_texts[i] for i in X_test_idx]

    combined_exclude_vocab = swap_words | _BROADER_RADIOLOGY_VOCAB
    vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), min_df=2,
        stop_words=list(combined_exclude_vocab),
    )
    Xtr = vectorizer.fit_transform(X_train_txt)
    Xte = vectorizer.transform(X_test_txt)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, y_train)
    y_prob = clf.predict_proba(Xte)[:, 1]
    y_pred = clf.predict(Xte)
    try:
        combined_vocab_auroc = float(roc_auc_score(y_test, y_prob))
    except ValueError:
        combined_vocab_auroc = None
    combined_vocab_acc = float(accuracy_score(y_test, y_pred))
    print(f"    n_features={len(vectorizer.get_feature_names_out())}  "
          f"acc={combined_vocab_acc:.4f}  auroc={combined_vocab_auroc}")

    # Reference number from Exp 8, loaded from its cached JSON if present,
    # so this experiment's "how much further did we drop" claim is anchored
    # to the SAME run rather than a hardcoded constant.
    exp8_path = out_dir / "exp8_closed_vocabulary_test.json"
    exp8_swap_excluded_auroc = None
    if exp8_path.exists():
        with open(exp8_path) as f:
            exp8_data = json.load(f)
        exp8_swap_excluded_auroc = exp8_data.get("swap_vocabulary_excluded", {}).get("auroc")

    near_chance_threshold = 0.60
    length_or_broader_vocab_explains_residual = (
        combined_vocab_auroc is not None and combined_vocab_auroc < near_chance_threshold
    )

    result = {
        "text_basis": (
            "9a/9b computed on SWAP_MAP-stripped text (matches Exp 8's "
            "'swap_vocabulary_excluded' condition, AUROC="
            f"{exp8_swap_excluded_auroc}); 9c additionally strips a broader "
            "radiology vocabulary list on top of SWAP_MAP."
        ),
        "9a_length_confound": {
            "mean_word_count_hallucinated": round(float(len_hallu.mean()), 3),
            "mean_word_count_faithful": round(float(len_faith.mean()), 3),
            "length_only_classifier_auroc": round(len_auroc, 4) if len_auroc is not None else None,
        },
        "9b_broader_vocab_confound": {
            "mean_broader_vocab_count_hallucinated": round(float(bc_hallu.mean()), 4),
            "mean_broader_vocab_count_faithful": round(float(bc_faith.mean()), 4),
            "broader_vocab_count_only_classifier_auroc": round(bc_auroc, 4) if bc_auroc is not None else None,
        },
        "9a_9b_combined_classifier_auroc": round(combo_auroc, 4) if combo_auroc is not None else None,
        "9c_full_classifier_with_swap_and_broader_vocab_excluded": {
            "n_vocab_excluded": len(combined_exclude_vocab),
            "accuracy": round(combined_vocab_acc, 4),
            "auroc": round(combined_vocab_auroc, 4) if combined_vocab_auroc is not None else None,
        },
        "exp8_reference_swap_only_excluded_auroc": exp8_swap_excluded_auroc,
        "additional_auroc_drop_from_broader_vocab_exclusion": (
            round(exp8_swap_excluded_auroc - combined_vocab_auroc, 4)
            if exp8_swap_excluded_auroc is not None and combined_vocab_auroc is not None else None
        ),
        "near_chance_threshold_used": near_chance_threshold,
        "length_or_broader_vocab_explains_residual": length_or_broader_vocab_explains_residual,
        "interpretation": (
            (
                "Excluding a BROADER radiology vocabulary (beyond just "
                "SWAP_MAP) drops AUROC close to chance. This means the "
                "residual signal Exp 8 found was a VOCABULARY-BREADTH "
                "confound, not a structural/length confound: eligible-for-"
                "injection reports simply use more anatomical/descriptive "
                "language in general (of which SWAP_MAP words are only a "
                "subset), and a text classifier can key on that broader "
                "vocabulary just as easily as on SWAP_MAP itself. The fix "
                "must therefore address WHICH reports are selected as "
                "candidates for injection (sampling / eligibility criteria "
                "in build_benchmark_multi.py), not just what word gets "
                "swapped once a report is selected."
            ) if length_or_broader_vocab_explains_residual else (
                "Even excluding a BROADER radiology vocabulary on top of "
                "SWAP_MAP, AUROC remains well above chance. Combined with "
                "the length-only and broader-vocab-count-only classifiers' "
                "own AUROCs (9a, 9b) as a diagnostic reference, this points "
                "toward a STRUCTURAL confound (report length/complexity "
                "itself, independent of specific vocabulary) as the more "
                "likely explanation for the residual leak, rather than a "
                "vocabulary-breadth issue alone. This is a harder problem: "
                "it means reports eligible for injection are systematically "
                "different in KIND (not just word choice) from reports that "
                "are not eligible, which a paraphrase-based fix targeting "
                "individual injected claims will NOT resolve -- the "
                "sampling/selection process for WHICH reports receive "
                "injections would need to be rebalanced (e.g. matching "
                "length/complexity between the clean and hallucinated "
                "pools) rather than just rewriting the injected text."
            )
        ),
    }
    json_path = out_dir / "exp9_eligibility_confound.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  → {json_path}")
    print(f"  length_or_broader_vocab_explains_residual: {length_or_broader_vocab_explains_residual}")

    return result


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 10 — Source-imbalance leak test (no API needed)
# ══════════════════════════════════════════════════════════════════════════
#
# Exp 9 ruled out length and broader-vocabulary confounds for the residual
# leak Exp 8 found (AUROC ~0.69 even with SWAP_MAP + broader vocab both
# excluded). The next cheap, obvious candidate: SAFER combines two
# datasets (OpenI + MIMIC-CXR) with likely different report-writing
# conventions (templating, punctuation, section headers, abbreviation
# style). If the hallucinated/faithful split is NOT perfectly balanced
# WITHIN each source (e.g. build_benchmark_multi.py's per-source injection
# happens to leave slightly different injection rates for OpenI vs
# MIMIC-CXR, or one source's reports are simply more likely to contain an
# invertible SWAP_MAP word and therefore over-represented in the
# hallucinated class), then `source` itself becomes an unintended,
# entirely-free label leak: a classifier doesn't need to read the report
# at all, it just needs to recognize "this looks like a MIMIC-CXR report"
# or "this looks like an OpenI report" via formatting/style alone.
#
# This experiment tests this directly and cheaply:
#   10a. Class-balance check: what fraction of each source's samples are
#        hallucinated? If this differs meaningfully between sources, that
#        IS the confound by definition -- no classifier needed to prove it,
#        just a cross-tab.
#   10b. A classifier using ONLY the one-hot source label as a feature (no
#        text at all) -- if this alone gets a non-trivial AUROC, source
#        imbalance is a real, exploitable leak on its own.
#   10c. Re-runs Exp 9's 9c classifier (SWAP_MAP + broader vocab excluded)
#        WITHIN each source separately (rather than pooled across both) --
#        if the pooled AUROC (0.69) drops substantially when computed
#        source-by-source, that confirms cross-source formatting
#        differences (not something intrinsic to either source's own
#        hallucinated-vs-faithful reports) were driving the pooled number.

def exp10_source_imbalance(root: Path, out_dir: Path, test_size: float = 0.25, seed: int = 42):
    print("\n" + "=" * 70)
    print("EXP 10: Source-imbalance leak test (no API needed)")
    print("=" * 70)

    if not HAS_SKLEARN:
        print("  [skip] scikit-learn not available")
        return None

    bench = load_benchmark(root)
    if bench is None:
        print("  [skip] benchmark file not found under data/v1/processed/")
        return None

    records = list(bench.values())
    for r in records:
        if r.get("source") is None:
            r["source"] = r["id"].split("_")[0] if "_" in r["id"] else "unknown"

    sources_present = sorted(set(r["source"] for r in records))
    if len(sources_present) < 2:
        print(f"  [skip] only one source found ({sources_present}) — this "
              f"experiment requires a multi-source benchmark")
        return None

    # ── 10a: class balance within each source ──
    print("\n  10a: Class balance (fraction hallucinated) within each source")
    balance_by_source = {}
    for src in sources_present:
        src_recs = [r for r in records if r["source"] == src]
        n = len(src_recs)
        n_hallu = sum(1 for r in src_recs if r.get("hallucinated"))
        frac = n_hallu / n if n else None
        balance_by_source[src] = {"n": n, "n_hallucinated": n_hallu,
                                   "frac_hallucinated": round(frac, 4) if frac is not None else None}
        print(f"    {src:<12} n={n:<6} n_hallucinated={n_hallu:<6} "
              f"frac_hallucinated={frac:.4f}" if frac is not None else f"    {src}: n=0")

    fracs = [v["frac_hallucinated"] for v in balance_by_source.values() if v["frac_hallucinated"] is not None]
    max_frac_gap = round(max(fracs) - min(fracs), 4) if len(fracs) >= 2 else None
    print(f"    max gap in hallucinated-fraction across sources: {max_frac_gap}")

    # ── 10b: source-only classifier (no text at all) ──
    print("\n  10b: Classifier using ONLY the source label (no report text)")
    y = np.array([1 if r.get("hallucinated") else 0 for r in records])
    source_onehot = pd.get_dummies(pd.Series([r["source"] for r in records])).values.astype(float)

    idx_train, idx_test, y_train, y_test = train_test_split(
        np.arange(len(records)), y, test_size=test_size, random_state=seed, stratify=y
    )
    src_train = source_onehot[idx_train]
    src_test = source_onehot[idx_test]

    src_clf = LogisticRegression(class_weight="balanced")
    src_clf.fit(src_train, y_train)
    src_pred_prob = src_clf.predict_proba(src_test)[:, 1]
    try:
        source_only_auroc = float(roc_auc_score(y_test, src_pred_prob))
    except ValueError:
        source_only_auroc = None
    print(f"    source-only classifier AUROC: {source_only_auroc}")

    # ── 10c: re-run Exp 9's 9c classifier WITHIN each source separately ──
    print("\n  10c: SWAP_MAP+broader-vocab-excluded classifier, computed WITHIN each source")

    swap_words = set()
    try:
        import sys as _sys
        for p in (root / "code/v1/01_data_prep", root / "01_data_prep"):
            if p.exists():
                _sys.path.insert(0, str(p))
        from importlib import import_module
        try:
            bb_mod = import_module("build_benchmark_multi")
        except ImportError:
            bb_mod = import_module("build_benchmark")
        for d in bb_mod.SWAP_MAP.values():
            swap_words.update(w.lower() for w in d.keys())
            swap_words.update(w.lower() for w in d.values())
    except Exception as e:
        print(f"    [warn] could not recover SWAP_MAP, skipping 10c: {e}")
        swap_words = None

    within_source_results = {}
    if swap_words is not None:
        combined_exclude_vocab = swap_words | _BROADER_RADIOLOGY_VOCAB
        for src in sources_present:
            src_recs = [r for r in records if r["source"] == src]
            src_texts = [r.get("report", "") for r in src_recs]
            src_labels = [1 if r.get("hallucinated") else 0 for r in src_recs]
            if sum(src_labels) < 10 or (len(src_labels) - sum(src_labels)) < 10:
                print(f"    [skip-source] {src}: not enough samples in one class")
                continue

            sX_train, sX_test, sy_train, sy_test = train_test_split(
                src_texts, src_labels, test_size=test_size, random_state=seed, stratify=src_labels
            )
            vectorizer = TfidfVectorizer(
                max_features=5000, ngram_range=(1, 2), min_df=2,
                stop_words=list(combined_exclude_vocab),
            )
            sXtr = vectorizer.fit_transform(sX_train)
            sXte = vectorizer.transform(sX_test)
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(sXtr, sy_train)
            sy_prob = clf.predict_proba(sXte)[:, 1]
            try:
                src_auroc = float(roc_auc_score(sy_test, sy_prob))
            except ValueError:
                src_auroc = None
            within_source_results[src] = {
                "n_train": len(sy_train), "n_test": len(sy_test),
                "auroc": round(src_auroc, 4) if src_auroc is not None else None,
            }
            print(f"    {src:<12} n_test={len(sy_test):<6} auroc={src_auroc}")

    # Reference the pooled Exp 9 number for a direct before/after
    exp9_path = out_dir / "exp9_eligibility_confound.json"
    exp9_pooled_auroc = None
    if exp9_path.exists():
        with open(exp9_path) as f:
            exp9_data = json.load(f)
        exp9_pooled_auroc = exp9_data.get(
            "9c_full_classifier_with_swap_and_broader_vocab_excluded", {}
        ).get("auroc")

    within_source_aurocs = [v["auroc"] for v in within_source_results.values() if v["auroc"] is not None]
    mean_within_source_auroc = round(float(np.mean(within_source_aurocs)), 4) if within_source_aurocs else None
    pooled_vs_within_drop = (
        round(exp9_pooled_auroc - mean_within_source_auroc, 4)
        if exp9_pooled_auroc is not None and mean_within_source_auroc is not None else None
    )

    source_leak_confirmed = (
        (max_frac_gap is not None and max_frac_gap > 0.05)
        or (source_only_auroc is not None and source_only_auroc > 0.60)
        or (pooled_vs_within_drop is not None and pooled_vs_within_drop > 0.10)
    )

    result = {
        "10a_class_balance_by_source": balance_by_source,
        "10a_max_hallucinated_fraction_gap_across_sources": max_frac_gap,
        "10b_source_only_classifier_auroc": round(source_only_auroc, 4) if source_only_auroc is not None else None,
        "10c_within_source_classifier_aurocs": within_source_results,
        "10c_mean_within_source_auroc": mean_within_source_auroc,
        "exp9_pooled_auroc_reference": exp9_pooled_auroc,
        "pooled_vs_mean_within_source_auroc_drop": pooled_vs_within_drop,
        "source_leak_confirmed": source_leak_confirmed,
        "interpretation": (
            (
                "Evidence supports a SOURCE-LEVEL confound: either the "
                "hallucinated/faithful split is imbalanced across sources "
                "(10a), a classifier using ONLY the source label achieves "
                "non-trivial AUROC (10b), and/or the pooled cross-source "
                "AUROC drops substantially when computed within each source "
                "separately (10c) -- meaning part of Exp 9's residual "
                "0.69 AUROC was driven by cross-source formatting/style "
                "differences rather than anything about the injected "
                "hallucination itself. The fix here is either (a) balancing "
                "injection rates exactly across sources, or (b) evaluating "
                "and reporting text-only-baseline leakage separately per "
                "source rather than pooled, since a pooled number can hide "
                "a source-driven artifact that doesn't reflect genuine "
                "hallucination-detection difficulty at all."
            ) if source_leak_confirmed else (
                "No strong evidence of a source-level confound: class "
                "balance is similar across sources (10a), a source-only "
                "classifier gets close to chance AUROC (10b), and within-"
                "source AUROC is close to the pooled Exp 9 number (10c). "
                "This means the residual leak identified in Exp 8/9 is NOT "
                "explained by combining two datasets with different "
                "reporting conventions -- it appears to be a genuine, "
                "source-independent property of the injection/selection "
                "process itself, and remains only partially diagnosed after "
                "four elimination rounds (grammar, closed vocabulary, "
                "broader vocabulary/length, and now source). This should be "
                "reported as an open, honestly-unresolved limitation of the "
                "benchmark's synthetic perturbation methodology rather than "
                "pursued further with additional cheap heuristic probes."
            )
        ),
    }
    json_path = out_dir / "exp10_source_imbalance.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  → {json_path}")
    print(f"  source_leak_confirmed: {source_leak_confirmed}")

    return result


# ══════════════════════════════════════════════════════════════════════════
# Report generation
# ══════════════════════════════════════════════════════════════════════════

def write_report(out_dir: Path, results: dict):
    lines = ["# SAFER Failure-Analysis Experiments — Summary Report", ""]
    lines.append(
        "Auto-generated from failure_analysis.py. Each section below reflects "
        "what was actually computed from your run's real outputs — sections "
        "for experiments that were skipped (missing input files) say so "
        "explicitly rather than being silently omitted."
    )
    lines.append("")

    exp_titles = {
        "exp1": "Experiment 1 — Discrimination-metric degenerate-classifier blind spot",
        "exp2": "Experiment 2 — MedGemma-vs-cluster disagreement / shortcut hypothesis",
        "exp3": "Experiment 3 — Per-source (OpenI vs MIMIC-CXR) generalization",
        "exp4": "Experiment 4 — HalluScore rank-instability driver analysis",
        "exp5": "Experiment 5 — Text-only leakage (trained classifier + span ablation)",
        "exp6": "Experiment 6 — Paraphrase-repair fix for the Exp 5 leak, re-validated",
        "exp7": "Experiment 7 — MCC-gated Discrimination fix validated on real models",
        "exp8": "Experiment 8 — Closed-vocabulary hypothesis test (no API needed)",
        "exp9": "Experiment 9 — Eligibility-confound test for the residual leak",
        "exp10": "Experiment 10 — Source-imbalance leak test",
    }

    for key, title in exp_titles.items():
        lines.append(f"## {title}")
        lines.append("")
        res = results.get(key)
        if res is None:
            lines.append("**SKIPPED** — required input files were not found. "
                         "See console output above for exactly which file was missing.")
        else:
            lines.append("```json")
            lines.append(json.dumps(res, indent=2)[:4000])
            lines.append("```")
        lines.append("")

    report_path = out_dir / "failure_analysis_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n[report] → {report_path}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SAFER failure-analysis experiments")
    parser.add_argument("--root", type=str, required=True,
                         help="Pipeline root containing data/ and results/ "
                              "(e.g. /mnt/ssd/users/durgesh/HEalthcare/Health2)")
    parser.add_argument("--only", nargs="+", default=None,
                         choices=["exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7", "exp8", "exp9", "exp10"],
                         help="Run only these experiments (default: all)")
    parser.add_argument("--exp6_use_llm", action="store_true",
                         help="EXP 6: use an LLM (Anthropic API) to paraphrase "
                              "injected claims instead of the default rule-based "
                              "grammar repair. Requires ANTHROPIC_API_KEY set in "
                              "your environment and the 'anthropic' package "
                              "installed. Makes real, billed API calls. Falls "
                              "back to rule-based repair automatically if the "
                              "key/package is missing rather than failing.")
    parser.add_argument("--exp6_llm_model", type=str, default="claude-sonnet-4-6",
                         help="Model string to use for --exp6_use_llm (default: claude-sonnet-4-6)")
    parser.add_argument("--exp6_llm_max_samples", type=int, default=None,
                         help="Cap the number of hallucinated samples sent to the "
                              "LLM (useful for a cheap pilot run before doing the "
                              "full benchmark). Default: no cap (all hallucinated "
                              "samples).")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = root / "results/v1/failure_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    to_run = args.only or ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7", "exp8", "exp9", "exp10"]
    results = {}

    if "exp1" in to_run:
        results["exp1"] = exp1_degenerate_blindspot(root, out_dir)
    if "exp2" in to_run:
        results["exp2"] = exp2_medgemma_disagreement(root, out_dir)
    if "exp3" in to_run:
        results["exp3"] = exp3_per_source_generalization(root, out_dir)
    if "exp4" in to_run:
        results["exp4"] = exp4_rank_instability(root, out_dir)
    if "exp5" in to_run:
        results["exp5"] = exp5_text_leakage(root, out_dir)
    if "exp6" in to_run:
        results["exp6"] = exp6_paraphrase_fix(
            root, out_dir, use_llm=args.exp6_use_llm,
            llm_model=args.exp6_llm_model, llm_max_samples=args.exp6_llm_max_samples,
        )
    if "exp7" in to_run:
        results["exp7"] = exp7_validate_mcc_gate(root, out_dir)
    if "exp8" in to_run:
        results["exp8"] = exp8_closed_vocabulary_test(root, out_dir)
    if "exp9" in to_run:
        results["exp9"] = exp9_eligibility_confound(root, out_dir)
    if "exp10" in to_run:
        results["exp10"] = exp10_source_imbalance(root, out_dir)

    write_report(out_dir, results)

    print("\n" + "=" * 70)
    print("DONE. All outputs under:", out_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
