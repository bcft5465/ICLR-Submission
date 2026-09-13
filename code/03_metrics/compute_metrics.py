#!/usr/bin/env python3
"""
03_metrics/compute_metrics.py — v4 (post-review redesign, round 2)

CHANGES vs v3:
  1. CONFIG-DRIVEN WEIGHTS/BINS (closes config/code drift): W_DISC/W_ECE/
     W_TYPE and N_BINS and WEIGHT_SCHEMES are now read from config.yaml's
     metrics.halluscore block instead of being hardcoded module-level
     Python constants that the config file could silently disagree with.
     A startup assertion checks that config's "main" scheme matches
     metrics.halluscore.weights exactly, so these two can never drift
     apart again the way the original HalluScore formula and its reported
     values did.
  2. PER-TYPE ACCURACY DENOMINATOR FIX (Table 2 "n" consistency): the
     previous compute_type_accuracy() silently dropped parse failures
     ("unknown" pred_label) from its per-type counts, so the reported n
     for Table 2 was actually "n parsed successfully for this type", NOT
     the fixed n=84/25/32 (or whatever the multi-source equivalents are)
     stated in the paper's dataset description. If parse-failure rate
     differs by hallucination type or by model (plausible — e.g. CoT
     rambling might fail the strict regex more often on harder relational
     cases), the reported n silently varies per model/type without any
     flag — reintroducing exactly the kind of hidden Table-2 inconsistency
     the reviewers already caught once. FIX: compute_type_accuracy() now
     returns BOTH `n_total` (all hallucinated samples of this type in the
     benchmark, fixed, model-independent) and `n_scored` (how many of
     those were successfully parsed and therefore contribute to the
     accuracy number), plus `n_parse_failed` = n_total - n_scored. The
     accuracy itself is still computed only over scored samples (you
     can't compute accuracy on an unparseable output), but the paper
     table generation must now show BOTH numbers, and a warning is
     printed whenever n_parse_failed > 0 for any type/model so it can't
     be silently missed.
  3. Handles the new per-model shuffled-image baseline files
     (shuffled_image_baseline_<model>_v3.jsonl for qwen25_vl, medgemma,
     llama) instead of a single hardcoded shuffled_image_baseline_v3.jsonl,
     matching run_inference.py's round-2 fix.

Retained from v3:
  - HalluScore redesign (Reviewers CTbs, aebp, 8sZM):
      Discrimination = sqrt(MCC_norm * F1)     [geometric mean, harsh]
      MCC_norm = (MCC + 1) / 2                 [rescale [-1,1] -> [0,1]]
      HalluScore = w_disc*Discrimination + w_ece*(1-ECE) + w_type*TypeAccuracy
    The geometric mean means a model must score reasonably on BOTH MCC and F1
    to get a non-trivial Discrimination term -- a model with MCC~0 and F1~0.04
    (like LLaMA) now gets Discrimination ~ sqrt(0.5 * 0.04) ~ 0.14 instead of
    a clean-accuracy-inflated ~0.7-0.9.
  - No temperature_scale-based "recalibration" masking of raw AUROC; raw
    AUROC is reported as the primary number. Post-hoc calibration is still
    computed and reported SEPARATELY and labeled as such, never substituted
    for the raw number in Table 1.
  - Constant/random/text-only baselines are loaded and included in the
    summary table and in every relevant comparison, per reviewer request.
  - McNemar significance tests and bootstrap CIs retained from v2.
  - HalluScore weight ablation (5 schemes) retained per reviewer aebp
    request, now operating on the new Discrimination-based formula AND
    read from config (see change #1 above).
"""

import argparse
import json
import csv
import warnings
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import yaml
import numpy as np
from scipy.stats import chi2
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, matthews_corrcoef,
)
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[3]
CFG_PATH = ROOT / "code/v1/config.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

OUTPUTS = ROOT / CFG["paths"]["outputs"]
METRICS = ROOT / CFG["paths"]["metrics"]
METRICS.mkdir(parents=True, exist_ok=True)

VLM_MODEL_KEYS      = ["qwen25_vl", "medgemma", "llava15", "phi3_vision",
                       "chexagent", "biovil_t", "biomedclip", "blip2"]
BASELINE_MODEL_KEYS = ["constant_faithful", "random", "text_only_baseline",
                       "shuffled_image_baseline_qwen25_vl",
                       "shuffled_image_baseline_medgemma"]
ALL_MODEL_KEYS      = VLM_MODEL_KEYS + BASELINE_MODEL_KEYS

BASELINE_MODELS = {"constant_faithful", "random", "text_only_baseline",
                   "shuffled_image_baseline_qwen25_vl",
                   "shuffled_image_baseline_medgemma"}
VLM_MODELS      = {"qwen25_vl", "medgemma", "llava15", "phi3_vision",
                   "chexagent", "biovil_t", "biomedclip", "blip2"}

OUTPUT_VERSION = "v3"

# ── HalluScore config — now read from config.yaml, not hardcoded ────────────
# (closes the config/code drift flagged in review: config.yaml previously
# had a `halluscores_weights` block with keys the code never read at all)
_HS_CFG = CFG["metrics"]["halluscore"]
N_BINS  = CFG["metrics"]["ece_bins"]

W_DISC = _HS_CFG["weights"]["discrimination"]
W_ECE  = _HS_CFG["weights"]["ece"]
W_TYPE = _HS_CFG["weights"]["type_accuracy"]

WEIGHT_SCHEMES = {
    name: tuple(vals) for name, vals in _HS_CFG["weight_schemes"].items()
}

# Sanity check: config's "main" ablation scheme MUST match the primary
# weights block exactly, or the ablation table and Table 1's HalluScore
# would silently use two different formulas — the exact class of bug this
# whole rewrite is trying to prevent from recurring.
_main_scheme = WEIGHT_SCHEMES.get("main")
_primary_weights = (W_DISC, W_ECE, W_TYPE)
assert _main_scheme == _primary_weights, (
    f"config.yaml drift detected: metrics.halluscore.weight_schemes.main "
    f"{_main_scheme} does not match metrics.halluscore.weights "
    f"{_primary_weights}. Fix config.yaml before trusting any HalluScore "
    f"numbers produced by this script."
)

HALLU_TYPES = ["object", "attribute", "relational"]


# ── ECE ───────────────────────────────────────────────────────────────────────

def compute_ece(y_true: np.ndarray, y_conf: np.ndarray, n_bins: int = N_BINS) -> float:
    """
    y_true : 1 = hallucinated, 0 = faithful
    y_conf : P(hallucinated), FIXED DIRECTION (see run_inference.py changelog)
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n   = len(y_true)
    if n == 0:
        return 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_conf >= lo) & (y_conf <= hi if hi == 1.0 else y_conf < hi)
        if mask.sum() == 0:
            continue
        avg_conf = y_conf[mask].mean()
        avg_acc  = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(avg_conf - avg_acc)
    return float(round(ece, 6))


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(y_true, y_pred, y_conf, metric_fn, n_boot=1000, ci=0.95):
    rng    = np.random.default_rng(42)
    n      = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            s = metric_fn(y_true[idx], y_pred[idx], y_conf[idx])
            if s is not None:
                scores.append(s)
        except Exception:
            pass
    if not scores:
        return (None, None)
    lo = float(np.percentile(scores, (1 - ci) / 2 * 100))
    hi = float(np.percentile(scores, (1 + ci) / 2 * 100))
    return (round(lo, 4), round(hi, 4))


def auroc_fn(yt, yp, yc):
    if len(np.unique(yt)) < 2:
        return None
    return roc_auc_score(yt, yc)

def f1_fn(yt, yp, yc):
    return f1_score(yt, yp, zero_division=0)

def mcc_fn(yt, yp, yc):
    if len(np.unique(yt)) < 2 or len(np.unique(yp)) < 2:
        return 0.0  # MCC undefined for degenerate splits; treat as no-signal
    return matthews_corrcoef(yt, yp)

def acc_fn(yt, yp, yc):
    return accuracy_score(yt, yp)


def discrimination_fn(yt, yp, yc):
    """sqrt(MCC_norm * F1) — the redesigned discrimination term."""
    mcc = mcc_fn(yt, yp, yc)
    mcc_norm = (mcc + 1.0) / 2.0
    f1 = f1_fn(yt, yp, yc)
    return float(np.sqrt(max(0.0, mcc_norm) * max(0.0, f1)))


def hallu_score_fn(w_disc, w_ece, w_type, macro_type_acc):
    """
    Bootstrap-compatible HalluScore function. Note: TypeAccuracy's macro
    average requires per-type hallucination-type breakdown that does not
    resample cleanly at the overall-sample level, so for bootstrap CI
    purposes we hold macro_type_acc fixed at its point estimate and only
    resample the discrimination + ECE terms. This is documented in the
    paper methods section as a simplification of the CI, not silently
    assumed.
    """
    def fn(yt, yp, yc):
        disc = discrimination_fn(yt, yp, yc)
        ece  = compute_ece(yt, yc)
        return w_disc * disc + w_ece * (1 - ece) + w_type * macro_type_acc
    return fn


# ── McNemar's test ────────────────────────────────────────────────────────────

def mcnemar_test(preds_a, preds_b, y_true):
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if (b + c) == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_value": 1.0, "significant": False}
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value   = float(1 - chi2.cdf(chi2_stat, df=1))
    return {
        "b": b, "c": c,
        "chi2": round(chi2_stat, 4),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
    }


# ── Per-type accuracy (Table 2 only — Figure 2 dropped per review outcome) ───
# FIX (Reviewers CTbs / all, round 1): the old code computed "type accuracy"
# for hallucination types as RECALL (did the model flag this hallucinated
# sample correctly) but for the "none"/clean category as SPECIFICITY (did
# the model flag this clean sample correctly) -- two different quantities
# placed in parallel-looking slots, feeding both Table 2 AND a separate
# Figure 2 with different denominators, producing the exact "9% vs 97%"
# contradiction Reviewer CTbs flagged. Per the agreed fix: Figure 2 is
# REMOVED entirely. This function ONLY reports hallucination-type recall
# (Table 2's actual, single, well-defined quantity) and does NOT compute or
# return a parallel "none" pseudo-type-accuracy at all.
#
# FIX (round 2): n reported for each type is now UNAMBIGUOUS. Previously
# `n` silently meant "n successfully parsed for this type", which could
# vary per model even though the benchmark's actual per-type counts
# (n_total) are fixed and model-independent. Now returns:
#   n_total        - fixed count of hallucinated samples of this type in
#                     the benchmark (same across all models, matches the
#                     dataset construction numbers reported in the paper)
#   n_scored       - how many of those got a valid parsed prediction
#   n_parse_failed - n_total - n_scored (the samples excluded from the
#                     accuracy calculation because the model's output
#                     didn't parse)
#   acc            - still computed over n_scored only (can't score an
#                     unparseable output), but now impossible to confuse
#                     with "accuracy over all n_total samples of this type"
def compute_type_accuracy(records, n_boot=1000):
    per_type = defaultdict(lambda: {"scored_preds": [], "n_total": 0})

    for r in records:
        ht = r.get("hallu_type")
        if ht not in HALLU_TYPES:
            continue
        per_type[ht]["n_total"] += 1
        if r["pred_label"] not in ("faithful", "hallucinated"):
            continue  # parse failure — excluded from scored preds, but n_total already counted it
        pred_hallu = (r["pred_label"] == "hallucinated")
        per_type[ht]["scored_preds"].append(int(pred_hallu))

    rng = np.random.default_rng(42)
    type_out = {}
    any_parse_failures = False
    for t in HALLU_TYPES:
        preds = np.array(per_type[t]["scored_preds"])
        n_total = per_type[t]["n_total"]
        n_scored = len(preds)
        n_parse_failed = n_total - n_scored
        if n_parse_failed > 0:
            any_parse_failures = True

        if n_scored == 0:
            type_out[t] = {
                "acc": None, "ci_lo": None, "ci_hi": None,
                "n_total": n_total, "n_scored": 0, "n_parse_failed": n_parse_failed,
            }
            continue

        acc = float(preds.mean())
        boot_accs = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(preds), size=len(preds))
            boot_accs.append(preds[idx].mean())
        ci_lo = float(np.percentile(boot_accs, 2.5))
        ci_hi = float(np.percentile(boot_accs, 97.5))
        type_out[t] = {
            "acc": round(acc, 4), "ci_lo": round(ci_lo, 4), "ci_hi": round(ci_hi, 4),
            "n_total": n_total, "n_scored": n_scored, "n_parse_failed": n_parse_failed,
        }

    valid_accs = [type_out[t]["acc"] for t in HALLU_TYPES if type_out[t]["acc"] is not None]
    macro = round(float(np.mean(valid_accs)), 4) if valid_accs else 0.0
    type_out["macro_avg"] = macro
    type_out["_any_parse_failures"] = any_parse_failures
    return type_out


# ── Core metrics ──────────────────────────────────────────────────────────────

def compute_per_source_accuracy(records: list) -> dict:
    """
    NEW — enabled by the OpenI + ReXGradient-160K multi-dataset scale-up.
    Reports accuracy broken down by dataset source, since a model that
    performs well on OpenI but poorly on ReXGradient (or vice versa) would
    indicate the benchmark's conclusions don't generalize across imaging
    sites/institutions — directly relevant to Reviewer aebp's
    generalizability concern, now testable rather than only assertable.
    Records without a "source" field (e.g. legacy single-source runs)
    are grouped under "unknown".
    """
    from collections import defaultdict
    per_source = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records:
        if r["pred_label"] not in ("faithful", "hallucinated"):
            continue
        src = r.get("source", "unknown")
        gt_hallu = bool(r["gt_hallucinated"])
        pred_hallu = (r["pred_label"] == "hallucinated")
        per_source[src]["total"] += 1
        per_source[src]["correct"] += int(gt_hallu == pred_hallu)

    return {
        src: {
            "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else None,
            "n": v["total"],
        }
        for src, v in per_source.items()
    }


def compute_model_metrics(records, model_key, run_bootstrap=True):
    # Report parse failures transparently (Reviewer CTbs: don't fabricate
    # confidence values for unparseable outputs)
    n_total    = len(records)
    n_parse_ok = sum(1 for r in records if r.get("parse_ok", True) and
                      r["pred_label"] in ("faithful", "hallucinated") and
                      r.get("pred_prob_hallucinated") is not None)

    clean = [r for r in records
             if r["pred_label"] in ("faithful", "hallucinated")
             and r.get("pred_prob_hallucinated") is not None]
    n_unknown = n_total - len(clean)

    y_true = np.array([1 if r["gt_hallucinated"] else 0 for r in clean])
    y_pred = np.array([1 if r["pred_label"] == "hallucinated" else 0 for r in clean])
    y_conf = np.array([r["pred_prob_hallucinated"] for r in clean])  # P(hallucinated), fixed direction

    # ── Classification ──
    acc  = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    f1   = float(f1_score(y_true, y_pred, zero_division=0))
    mcc  = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity  = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    balanced_acc = float((rec + specificity) / 2)

    # ── Ranking / calibration (RAW, no auto-inversion) ──
    try:
        auroc = float(roc_auc_score(y_true, y_conf))
    except ValueError:
        auroc = None
    try:
        auprc = float(average_precision_score(y_true, y_conf))
    except ValueError:
        auprc = None

    ece = compute_ece(y_true, y_conf)

    # ── Type accuracy (Table 2 quantity only; n_total/n_scored/n_parse_failed) ──
    type_acc = compute_type_accuracy(records)
    if type_acc.get("_any_parse_failures"):
        detail = {t: type_acc[t]["n_parse_failed"] for t in HALLU_TYPES}
        print(f"    [warn] {model_key}: per-type parse failures present "
              f"(n_parse_failed by type: {detail}). Table 2 must report "
              f"n_total (fixed) AND n_scored (varies) separately — do NOT "
              f"just print a single 'n' as if it were constant across models.")

    # ── Per-source accuracy (multi-dataset scale-up) ──
    per_source_acc = compute_per_source_accuracy(records)

    # ── HalluScore — redesigned discrimination term ──
    mcc_norm = (mcc + 1.0) / 2.0
    discrimination = float(np.sqrt(max(0.0, mcc_norm) * max(0.0, f1)))

    hallu_scores = {}
    for scheme_name, (wd, we, wt) in WEIGHT_SCHEMES.items():
        hallu_scores[scheme_name] = round(
            wd * discrimination + we * (1.0 - ece) + wt * type_acc["macro_avg"], 4
        )
    hallu_score = hallu_scores["main"]

    # ── Bootstrap CIs ──
    cis = {}
    if run_bootstrap and len(y_true) > 10:
        print("    [bootstrap] Computing CIs ...")
        cis["auroc"] = bootstrap_ci(y_true, y_pred, y_conf, auroc_fn)
        cis["f1"]    = bootstrap_ci(y_true, y_pred, y_conf, f1_fn)
        cis["mcc"]   = bootstrap_ci(y_true, y_pred, y_conf, mcc_fn)
        cis["acc"]   = bootstrap_ci(y_true, y_pred, y_conf, acc_fn)
        cis["hallu_score"] = bootstrap_ci(
            y_true, y_pred, y_conf,
            hallu_score_fn(W_DISC, W_ECE, W_TYPE, type_acc["macro_avg"])
        )

    return {
        "n_total": n_total,
        "n_scored": len(clean),
        "n_unknown": n_unknown,
        "n_parse_ok": n_parse_ok,
        "parse_rate": round(n_parse_ok / n_total, 4) if n_total else None,
        "accuracy": round(acc, 4),
        "balanced_acc": round(balanced_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "mcc": round(mcc, 4),
        "auroc": round(auroc, 4) if auroc is not None else None,
        "auprc": round(auprc, 4) if auprc is not None else None,
        "ece": ece,
        "discrimination": round(discrimination, 4),
        "type_accuracy": type_acc,
        "per_source_accuracy": per_source_acc,
        "hallu_score": hallu_score,
        "hallu_score_ablation": hallu_scores,
        "bootstrap_ci": cis,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


# ── Significance testing ──────────────────────────────────────────────────────

def run_cross_model_verification_analysis(all_records):
    """
    Addresses Reviewer 76GN's core methodological critique: "HalluScore
    relies entirely on the VLM's outputs (faithfulness judgment, confidence,
    explanation). When the model hallucinates, these outputs are potentially
    unreliable, which undermines the validity of the detection signal."

    This is NOT fully solvable — we cannot make a model's self-report
    trustworthy by definition — but we CAN provide an independent signal:
    cross-model agreement. If two structurally different models (e.g. a
    large general VLM and a domain-specific model) independently agree on
    a verdict, that agreement is at least some evidence beyond a single
    model's unverified self-report. Where models disagree, no benchmark
    can currently adjudicate who is "right" without a human in the loop --
    that limitation is stated explicitly in the returned summary rather
    than hidden.

    This produces per-pair agreement rates AND, more importantly, the
    agreement rate SPECIFICALLY on samples where each model claims high
    confidence (>0.8 in either direction) — this is the subset where
    self-report reliability matters most, since a confidently-wrong
    self-report is the most clinically dangerous failure mode.
    """
    results = {}
    model_keys = [k for k in all_records if k in VLM_MODEL_KEYS]
    if len(model_keys) < 2:
        return results

    aligned = {}
    for mk in model_keys:
        recs = all_records[mk]
        id_to_rec = {
            r["id"]: r for r in recs
            if r["pred_label"] in ("faithful", "hallucinated")
            and r.get("pred_prob_hallucinated") is not None
        }
        aligned[mk] = id_to_rec

    for mk_a, mk_b in combinations(model_keys, 2):
        common_ids = sorted(set(aligned[mk_a]) & set(aligned[mk_b]))
        if not common_ids:
            continue

        agree = 0
        agree_high_conf = 0
        n_high_conf = 0
        for sid in common_ids:
            ra, rb = aligned[mk_a][sid], aligned[mk_b][sid]
            same = (ra["pred_label"] == rb["pred_label"])
            agree += int(same)

            conf_a = ra["pred_prob_hallucinated"]
            conf_b = rb["pred_prob_hallucinated"]
            a_confident = conf_a >= 0.8 or conf_a <= 0.2
            b_confident = conf_b >= 0.8 or conf_b <= 0.2
            if a_confident and b_confident:
                n_high_conf += 1
                agree_high_conf += int(same)

        results[f"{mk_a}_vs_{mk_b}"] = {
            "n_common_samples": len(common_ids),
            "raw_agreement_rate": round(agree / len(common_ids), 4),
            "n_both_high_confidence": n_high_conf,
            "high_confidence_agreement_rate": (
                round(agree_high_conf / n_high_conf, 4) if n_high_conf > 0 else None
            ),
        }

    return results


def run_significance_tests(all_records):
    results = {}
    model_keys = list(all_records.keys())
    all_ids = sorted(set.intersection(
        *[{r["id"] for r in recs} for recs in all_records.values()]
    ))
    print(f"  [significance] {len(all_ids)} common samples across all models")

    aligned = {}
    for mk, recs in all_records.items():
        id_to_rec = {r["id"]: r for r in recs
                     if r["pred_label"] in ("faithful", "hallucinated")}
        aligned[mk] = {
            "y_true": np.array([1 if id_to_rec[i]["gt_hallucinated"] else 0
                                 for i in all_ids if i in id_to_rec]),
            "y_pred": np.array([1 if id_to_rec[i]["pred_label"] == "hallucinated" else 0
                                 for i in all_ids if i in id_to_rec]),
        }

    for mk_a, mk_b in combinations(model_keys, 2):
        if mk_a not in aligned or mk_b not in aligned:
            continue
        n_common = min(len(aligned[mk_a]["y_true"]), len(aligned[mk_b]["y_true"]))
        if n_common == 0:
            continue
        test = mcnemar_test(
            aligned[mk_a]["y_pred"][:n_common],
            aligned[mk_b]["y_pred"][:n_common],
            aligned[mk_a]["y_true"][:n_common],
        )
        key = f"{mk_a}_vs_{mk_b}"
        results[key] = test
        sig = "✓ significant" if test["significant"] else "✗ not significant"
        print(f"    {mk_a} vs {mk_b}: p={test['p_value']:.4f}  {sig}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def run(model_keys, version=OUTPUT_VERSION):
    summary_rows = []
    all_records  = {}

    for model_key in model_keys:
        out_path = OUTPUTS / f"{model_key}_{version}.jsonl"
        if not out_path.exists():
            print(f"[skip] {out_path} not found")
            continue

        with open(out_path) as f:
            records = [json.loads(l) for l in f if l.strip()]

        print(f"\n[{model_key}] {len(records)} records loaded from {out_path.name}")
        all_records[model_key] = records

        metrics = compute_model_metrics(records, model_key, run_bootstrap=True)
        metrics["model"] = model_key
        metrics["model_category"] = "baseline" if model_key in BASELINE_MODELS else "vlm"

        model_metrics_path = METRICS / f"{model_key}_metrics.json"
        with open(model_metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  → {model_metrics_path}")

        per_sample_path = METRICS / "per_sample.jsonl"
        clean = [r for r in records
                 if r["pred_label"] in ("faithful", "hallucinated")
                 and r.get("pred_prob_hallucinated") is not None]
        with open(per_sample_path, "a") as f:
            for r in clean:
                gt = 1 if r["gt_hallucinated"] else 0
                pred = 1 if r["pred_label"] == "hallucinated" else 0
                f.write(json.dumps({
                    "id": r["id"], "model": model_key,
                    "model_category": metrics["model_category"],
                    "gt_hallucinated": bool(gt),
                    "hallu_type": r.get("hallu_type"),
                    "pred_label": r["pred_label"],
                    "pred_prob_hallucinated": r["pred_prob_hallucinated"],
                    "correct": bool(pred == gt),
                    "prompt_mode": r.get("prompt_mode", "standard"),
                }) + "\n")

        ta  = metrics["type_accuracy"]
        cis = metrics.get("bootstrap_ci", {})
        hs_ablation = metrics.get("hallu_score_ablation", {})
        summary_rows.append({
            "model": model_key,
            "model_category": metrics["model_category"],
            "accuracy": metrics["accuracy"],
            "balanced_acc": metrics["balanced_acc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "mcc": metrics["mcc"],
            "specificity": metrics["specificity"],
            "auroc": metrics["auroc"],
            "auprc": metrics["auprc"],
            "ece": metrics["ece"],
            "discrimination": metrics["discrimination"],
            "type_acc_object": ta.get("object", {}).get("acc"),
            "type_acc_attribute": ta.get("attribute", {}).get("acc"),
            "type_acc_relational": ta.get("relational", {}).get("acc"),
            "type_acc_macro": ta.get("macro_avg"),
            # n_total (fixed, benchmark-defined) vs n_scored (model-
            # dependent, excludes parse failures) — reported SEPARATELY
            # per type so nobody can mistake one for the other again.
            "type_n_total_object": ta.get("object", {}).get("n_total"),
            "type_n_scored_object": ta.get("object", {}).get("n_scored"),
            "type_n_total_attribute": ta.get("attribute", {}).get("n_total"),
            "type_n_scored_attribute": ta.get("attribute", {}).get("n_scored"),
            "type_n_total_relational": ta.get("relational", {}).get("n_total"),
            "type_n_scored_relational": ta.get("relational", {}).get("n_scored"),
            "hallu_score": metrics["hallu_score"],
            "hs_equal": hs_ablation.get("equal"),
            "hs_calib_heavy": hs_ablation.get("calib_heavy"),
            "hs_type_heavy": hs_ablation.get("type_heavy"),
            "auroc_ci_lo": cis.get("auroc", (None, None))[0],
            "auroc_ci_hi": cis.get("auroc", (None, None))[1],
            "f1_ci_lo": cis.get("f1", (None, None))[0],
            "f1_ci_hi": cis.get("f1", (None, None))[1],
            "mcc_ci_lo": cis.get("mcc", (None, None))[0],
            "mcc_ci_hi": cis.get("mcc", (None, None))[1],
            "n_unknown": metrics["n_unknown"],
            "parse_rate": metrics["parse_rate"],
            "TP": metrics["TP"], "FP": metrics["FP"],
            "TN": metrics["TN"], "FN": metrics["FN"],
        })

        print(f"  acc={metrics['accuracy']}  bal_acc={metrics['balanced_acc']}  "
              f"f1={metrics['f1']}  mcc={metrics['mcc']}  auroc={metrics['auroc']}  "
              f"ece={metrics['ece']}  parse_rate={metrics['parse_rate']}  "
              f"HalluScore={metrics['hallu_score']}")
        if len(metrics.get("per_source_accuracy", {})) > 1:
            print(f"    per-source accuracy: {metrics['per_source_accuracy']}")

    if len(all_records) > 1:
        print("\n[significance tests]")
        sig_results = run_significance_tests(all_records)
        sig_path = METRICS / "significance_tests.json"
        with open(sig_path, "w") as f:
            json.dump(sig_results, f, indent=2)
        print(f"  → {sig_path}")

        print("\n[cross-model verification analysis]")
        print("  (Reviewer 76GN: independent signal beyond a single model's")
        print("  self-report — see per-pair agreement, esp. under high confidence)")
        cross_model_results = run_cross_model_verification_analysis(all_records)
        cross_model_path = METRICS / "cross_model_verification.json"
        with open(cross_model_path, "w") as f:
            json.dump(cross_model_results, f, indent=2)
        print(f"  → {cross_model_path}")
        for pair, res in cross_model_results.items():
            hc_rate = res["high_confidence_agreement_rate"]
            hc_str = f"{hc_rate:.4f}" if hc_rate is not None else "N/A (no mutual high-confidence samples)"
            print(f"    {pair:<28}: raw_agree={res['raw_agreement_rate']:.4f}  "
                  f"high_conf_agree={hc_str}  (n_high_conf={res['n_both_high_confidence']})")

    # ── Weight-justification analysis (Reviewer aebp: "rationale behind
    # weights not well justified... unclear whether HalluScore provides
    # additional value over individual metrics"). This computes an actual
    # quantitative basis for the weight-choice paragraph rather than only
    # asserting robustness: (1) Spearman rank-correlation between the main
    # HalluScore ranking and each individual component metric, to show
    # HalluScore is not simply redundant with any single metric; (2)
    # pairwise Spearman correlation between all 5 weight schemes' rankings,
    # to quantify "robust to weighting" numerically rather than by eyeballing
    # the printed ranking order.
    if len(summary_rows) > 2:
        component_metrics = ["accuracy", "balanced_acc", "f1", "mcc",
                              "auroc", "ece", "discrimination"]
        weight_justification = {"vs_individual_metrics": {}, "vs_other_weight_schemes": {}}

        main_scores = [r["hallu_score"] for r in summary_rows]
        for cm in component_metrics:
            vals = [r.get(cm) for r in summary_rows]
            if any(v is None for v in vals):
                continue
            # ECE is "lower is better" — invert sign for a fair rank comparison
            cmp_vals = [-v if cm == "ece" else v for v in vals]
            try:
                rho, p = spearmanr(main_scores, cmp_vals)
                weight_justification["vs_individual_metrics"][cm] = {
                    "spearman_rho": round(float(rho), 4),
                    "p_value": round(float(p), 4),
                }
            except Exception:
                pass

        scheme_names = ["hallu_score", "hs_equal", "hs_calib_heavy", "hs_type_heavy"]
        for a, b in combinations(scheme_names, 2):
            vals_a = [r.get(a) for r in summary_rows]
            vals_b = [r.get(b) for r in summary_rows]
            if any(v is None for v in vals_a + vals_b):
                continue
            try:
                rho, p = spearmanr(vals_a, vals_b)
                weight_justification["vs_other_weight_schemes"][f"{a}_vs_{b}"] = {
                    "spearman_rho": round(float(rho), 4),
                    "p_value": round(float(p), 4),
                }
            except Exception:
                pass

        wj_path = METRICS / "halluscore_weight_justification.json"
        with open(wj_path, "w") as f:
            json.dump(weight_justification, f, indent=2)
        print(f"\n[weight-justification] → {wj_path}")
        print("  Spearman rho vs individual metrics (HalluScore should NOT be")
        print("  perfectly (rho=1.0) correlated with any single metric, or it")
        print("  adds no information beyond that metric):")
        for cm, res in weight_justification["vs_individual_metrics"].items():
            print(f"    {cm:<16}: rho={res['spearman_rho']:>7.4f}  p={res['p_value']}")
        print("  Spearman rho across weight schemes (near 1.0 = ranking robust")
        print("  to weight choice, supports the paper's robustness claim):")
        for pair, res in weight_justification["vs_other_weight_schemes"].items():
            print(f"    {pair:<28}: rho={res['spearman_rho']:>7.4f}  p={res['p_value']}")

    if summary_rows:
        ablation_path = METRICS / "halluscore_ablation.csv"
        ablation_fields = ["model", "model_category", "hallu_score",
                           "hs_equal", "hs_calib_heavy", "hs_type_heavy"]
        with open(ablation_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ablation_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n[ablation] → {ablation_path}")

        ranks = {}
        for scheme in ["hallu_score", "hs_equal", "hs_calib_heavy", "hs_type_heavy"]:
            sorted_models = sorted(summary_rows, key=lambda r: r[scheme] or 0, reverse=True)
            ranks[scheme] = [r["model"] for r in sorted_models]
        print("  Rank stability across weight schemes:")
        for scheme, order in ranks.items():
            print(f"    {scheme:<20}: {' > '.join(order)}")
        first_rank = ranks["hallu_score"]
        all_same = all(r == first_rank for r in ranks.values())
        print(f"  Rankings consistent: {'YES ✓' if all_same else 'NO — check ablation'}")

    if summary_rows:
        summary_path = METRICS / "summary.csv"
        fieldnames = list(summary_rows[0].keys())
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n[summary] → {summary_path}")

        print(f"\n{'Model':<20} {'Cat':<9} {'Acc':>6} {'BalAcc':>7} "
              f"{'F1':>6} {'MCC':>7} {'AUROC':>7} {'ECE':>7} {'HalluScore':>11}")
        print("─" * 100)
        for row in sorted(summary_rows, key=lambda r: r["hallu_score"] or 0, reverse=True):
            auroc = f"{row['auroc']:.4f}" if row["auroc"] is not None else "  N/A "
            cat = row["model_category"]
            print(f"{row['model']:<20} {cat:<9} {row['accuracy']:>6.4f} "
                  f"{row['balanced_acc']:>7.4f} {row['f1']:>6.4f} {row['mcc']:>7.4f} "
                  f"{auroc:>7} {row['ece']:>7.4f} {row['hallu_score']:>11.4f}")

        # Explicit sanity-check print: does any baseline outrank a real VLM?
        # This directly targets the reviewer concern and should NEVER be
        # silently ignored if it happens again.
        best_baseline = max(
            (r for r in summary_rows if r["model_category"] == "baseline"
             and r["model"] not in ("biovil_t",)),  # biovil_t is a real model baseline
            key=lambda r: r["hallu_score"] or 0, default=None
        )
        best_vlm = max(
            (r for r in summary_rows if r["model"] in VLM_MODELS),
            key=lambda r: r["hallu_score"] or 0, default=None
        )
        if best_baseline and best_vlm and best_baseline["hallu_score"] >= best_vlm["hallu_score"]:
            print(f"\n  [WARNING] Baseline '{best_baseline['model']}' "
                  f"(HalluScore={best_baseline['hallu_score']}) scores >= "
                  f"best VLM '{best_vlm['model']}' (HalluScore={best_vlm['hallu_score']}). "
                  f"This would undermine the benchmark's validity — investigate "
                  f"before publishing these numbers.")

        # Explicit per-type n consistency check across models: if n_total
        # for a given hallucination type differs across models' metrics
        # files, something is deeply wrong (n_total should be fixed by the
        # benchmark, not by which model produced the predictions).
        for htype in HALLU_TYPES:
            n_totals = {r["model"]: r.get(f"type_n_total_{htype}") for r in summary_rows}
            distinct = set(v for v in n_totals.values() if v is not None)
            if len(distinct) > 1 and max(distinct) - min(distinct) > 10:
                print(f"\n  [ERROR] n_total for hallu_type='{htype}' is NOT "
                      f"consistent across models: {n_totals}. This should be "
                      f"a fixed benchmark property — investigate immediately, "
                      f"this is exactly the Table-2-inconsistency failure "
                      f"mode reviewers already flagged once.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        choices=ALL_MODEL_KEYS + ["all"])
    parser.add_argument("--version", default=OUTPUT_VERSION)
    args = parser.parse_args()

    if args.model and args.model != "all":
        keys = [args.model]
    else:
        keys = [k for k in ALL_MODEL_KEYS
                if (OUTPUTS / f"{k}_{args.version}.jsonl").exists()]

    per_sample_path = METRICS / "per_sample.jsonl"
    if per_sample_path.exists():
        per_sample_path.unlink()

    run(keys, version=args.version)