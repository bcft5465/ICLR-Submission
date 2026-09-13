"""
01_data_prep/build_benchmark.py — v3 (post-review, LEGACY / non-default path)

*** THIS SCRIPT IS NO LONGER CALLED BY run_pipeline.sh BY DEFAULT. ***
The default benchmark build is now build_benchmark_multi.py (OpenI +
ReXGradient-160K). This script is kept ONLY so the original 500-sample
OpenI-only submission numbers can be exactly reproduced on request (e.g.
if a reviewer specifically asks to see the original single-source run,
or for an ablation showing "does the multi-source scale-up change
conclusions vs. the original small benchmark"). Run it explicitly:

    python build_benchmark.py

It reads its sample size from config.yaml's `dataset.legacy_single_source`
block (n_samples: 500), NOT from `dataset.n_total` / `dataset.sources`,
which now belong to the multi-source default path. This separation is
deliberate: the two configs must not be able to silently drift into
meaning the same thing or overriding each other.

REVIEWER-FLAGGED LIMITATION (documented here, not silently patched away):
Reviewer CTbs and Reviewer 8sZM both raise that hallucinations are produced
by a FIXED, small, closed-vocabulary word-swap dictionary (SWAP_MAP below,
e.g. "left"<->"right", "large"<->"small"). This creates two distinct risks:

  1. A text-only classifier could learn to detect the antonym-swap PATTERN
     itself (a linguistic artifact) without ever using the image. This is
     exactly why a text-only baseline is now REQUIRED — see
     02_inference/run_inference.py: run_text_only_baseline(), and its
     results in the summary table produced by 03_metrics/compute_metrics.py.
     If that baseline scores well above chance, it is direct evidence the
     benchmark leaks signal through text artifacts, and Table 1 / the paper
     text must say so explicitly rather than omit it.

  2. Fixed, manually-perturbed claims are not representative of the
     diverse, naturally-occurring hallucinations a real deployed VLM would
     produce under varied prompts, decoding settings, and seeds (Reviewer
     8sZM). Per that review's explicit recommendation, the paper's scope
     claims should be narrowed to "image-claim verification on
     synthetically perturbed claims" rather than general "hallucination
     detection in radiology," unless/until a naturally-generated-
     hallucination subset is added (e.g. sampling real VLM outputs and
     having radiologists label them, rather than only injecting fixed
     swaps). THIS IS STILL NOT DONE in either build script — it remains an
     open limitation, not silently expanded here.

This file's injection LOGIC is unchanged from v1/v2 — SWAP_MAP is exposed
at module level so run_inference.py's run_text_only_baseline() can import
and test against it directly, matching the equivalent SWAP_MAP in
build_benchmark_multi.py (kept identical between both scripts deliberately;
if you change one, change both, or better, factor it into a shared module —
see note at bottom of file).
"""

import json, random, re
from pathlib import Path

import pandas as pd
import yaml

# ── Paths ────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[3]
CFG_PATH = ROOT / "code/v1/config.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

RAW       = ROOT / CFG["paths"]["data_raw"]
PROCESSED = ROOT / CFG["paths"]["data_processed"]
PROCESSED.mkdir(parents=True, exist_ok=True)

IMAGES_DIR = RAW / "images" / "images_normalized"
REPORTS_CSV = RAW / "indiana_reports.csv"
PROJ_CSV    = RAW / "indiana_projections.csv"

LEGACY_CFG = CFG["dataset"]["legacy_single_source"]
SEED     = CFG["dataset"]["random_seed"]   # shared seed key, same value as multi-source path
N        = LEGACY_CFG["n_samples"]
INJ_RATE = CFG["dataset"]["injection_rate"]  # shared injection_rate, same as multi-source path
random.seed(SEED)

# ── Hallucination vocabulary ─────────────────────────────────
# NOTE: this is a small, closed vocabulary — see module docstring above for
# the reviewer-flagged risk this creates and the text-only baseline that
# now tests for it directly. Kept IDENTICAL to build_benchmark_multi.py's
# SWAP_MAP on purpose (same hallucination-injection semantics regardless
# of which dataset script produced the benchmark file).
SWAP_MAP = {
    "object": {
        "lung": "liver", "heart": "kidney", "pleura": "diaphragm",
        "aorta": "trachea", "rib": "clavicle", "spine": "sternum",
    },
    "attribute": {
        "large": "small", "small": "large",
        "increased": "decreased", "decreased": "increased",
        "bilateral": "unilateral", "unilateral": "bilateral",
        "mild": "severe", "severe": "mild",
    },
    "relational": {
        "left": "right", "right": "left",
        "upper": "lower", "lower": "upper",
        "anterior": "posterior", "posterior": "anterior",
    },
}

def inject(text: str, htype: str) -> tuple:
    for orig, fake in SWAP_MAP[htype].items():
        pattern = re.compile(re.escape(orig), re.IGNORECASE)
        if pattern.search(text):
            new_text = pattern.sub(fake, text, count=1)
            return new_text, orig, fake
    return text, "", ""

# ── Load & merge CSVs ────────────────────────────────────────
# DATA-HANDLING POLICY (Reviewer CTbs asked explicitly: "explain how images,
# patients, uncertain findings, and multiple radiograph views were handled."
# This is answered concretely here, not just described in prose elsewhere,
# so the answer stays correct if the loading logic ever changes.)
#
#   Multiple views per patient/study: OpenI Indiana frequently provides
#   BOTH a frontal and a lateral projection per study. We deliberately keep
#   ONLY the "Frontal" projection per uid (one image per uid) and DROP all
#   lateral views. This means: (a) each benchmark sample has exactly one
#   image, avoiding any ambiguity about which view a claim refers to, and
#   (b) any finding in the report that is only visible on the (discarded)
#   lateral view is NOT verifiable from the image the model receives. This
#   is a real limitation — a small number of report findings may be
#   ungroundable from the frontal image alone — and should be stated in
#   the paper's Limitations section verbatim from this comment.
#
#   Multiple patients/studies: each row is keyed by a unique study "uid"
#   from the Indiana dataset; no cross-patient merging occurs. If the same
#   patient has multiple studies over time, each study-uid is treated as an
#   independent sample (patient identity is not tracked or deduplicated
#   across studies). This means the benchmark could in principle contain
#   more than one sample from the same underlying patient. This is now
#   measured explicitly by report_loading_stats() below.
#
#   Uncertain findings: the source reports frequently contain hedging
#   language (e.g., "possible", "cannot exclude", "likely represents").
#   NO special handling is applied to hedged findings — they are extracted
#   and potentially perturbed identically to unhedged findings. This means
#   a "hallucinated" claim could, in rare cases, be a perturbation of an
#   already-uncertain finding, which is a softer ground truth than a
#   perturbation of a clearly-stated finding. This is not currently
#   filtered or flagged per-sample; doing so is documented here as a
#   concrete direction for future benchmark versions rather than silently
#   ignored.
def load_data() -> list[dict]:
    reports = pd.read_csv(REPORTS_CSV)
    proj    = pd.read_csv(PROJ_CSV)

    frontal = proj[proj["projection"] == "Frontal"].copy()
    frontal["img_path"] = frontal["filename"].apply(
        lambda f: str(IMAGES_DIR / f)
    )
    frontal = frontal[frontal["img_path"].apply(lambda p: Path(p).exists())]

    merged = frontal.merge(reports[["uid", "findings", "impression"]], on="uid")

    records = []
    for _, row in merged.iterrows():
        findings   = str(row.get("findings",   "") or "").strip()
        impression = str(row.get("impression", "") or "").strip()
        report = f"{findings} {impression}".strip()
        if len(report) < 20:
            continue
        records.append({
            "id":       str(row["uid"]),
            "img_path": row["img_path"],
            "report":   report,
        })
    return records


HEDGE_TERMS = [
    "possible", "possibly", "cannot exclude", "cannot be excluded",
    "likely represents", "likely reflects", "may represent", "may reflect",
    "suggestive of", "cannot rule out", "differential includes",
    "uncertain", "equivocal", "borderline",
]

def report_loading_stats(records: list[dict]) -> dict:
    """
    Emits concrete counts answering Reviewer CTbs's question, computed from
    the ACTUAL loaded data rather than asserted in prose. Call this after
    load_data() and print/log the result; include these numbers directly
    in the paper's Data Construction section.
    """
    proj = pd.read_csv(PROJ_CSV)
    n_frontal_kept = len(records)
    n_lateral_dropped = int((proj["projection"] != "Frontal").sum())

    n_hedged = sum(
        1 for r in records
        if any(term in r["report"].lower() for term in HEDGE_TERMS)
    )

    ids = [r["id"] for r in records]
    n_unique_studies = len(set(ids))
    n_duplicate_study_rows = len(ids) - n_unique_studies

    return {
        "n_frontal_samples_kept": n_frontal_kept,
        "n_lateral_view_rows_dropped": n_lateral_dropped,
        "n_reports_with_hedge_language": n_hedged,
        "pct_reports_with_hedge_language": round(n_hedged / max(n_frontal_kept, 1) * 100, 1),
        "n_unique_study_ids": n_unique_studies,
        "n_duplicate_study_id_rows": n_duplicate_study_rows,
        "note": ("Patient-level deduplication is NOT performed — "
                 "n_unique_study_ids is the correct denominator for "
                 "'distinct studies', not necessarily 'distinct patients'."),
    }

# ── Build benchmark ──────────────────────────────────────────
def build(records: list[dict]) -> list[dict]:
    random.shuffle(records)
    records = records[:N]

    n_hallu = int(len(records) * INJ_RATE)
    htypes  = (["object", "attribute", "relational"] * n_hallu)[:n_hallu]
    random.shuffle(htypes)

    benchmark = []
    for i, rec in enumerate(records):
        entry = {
            "id":              rec["id"],
            "img_path":        rec["img_path"],
            "original_report": rec["report"],
            "hallucinated":    False,
            "hallu_type":      None,
            "original_span":   None,
            "injected_span":   None,
            "report":          rec["report"],
        }
        if i < n_hallu:
            htype = htypes[i]
            new_text, orig, fake = inject(rec["report"], htype)
            if orig:
                entry.update({
                    "hallucinated":  True,
                    "hallu_type":    htype,
                    "original_span": orig,
                    "injected_span": fake,
                    "report":        new_text,
                })
        benchmark.append(entry)
    return benchmark

# ── Save ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[LEGACY PATH] This is the original 500-sample OpenI-only builder.")
    print("  The default pipeline now uses build_benchmark_multi.py instead.")
    print(f"  n_samples (from config.yaml dataset.legacy_single_source) = {N}")
    print()

    print("[load] Reading CSVs ...")
    records = load_data()
    print(f"  → {len(records)} valid paired records")

    stats = report_loading_stats(records)
    stats_path = PROCESSED / "data_handling_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[stats] Data-handling policy stats → {stats_path}")
    for k, v in stats.items():
        print(f"    {k}: {v}")
    print("  [reminder] Report these numbers verbatim in the paper's Data "
          "Construction / Limitations section (Reviewer CTbs asked for this "
          "explicitly).")

    print("[build] Injecting hallucinations ...")
    benchmark = build(records)

    out_path = PROCESSED / "benchmark_v1.jsonl"
    with open(out_path, "w") as f:
        for entry in benchmark:
            f.write(json.dumps(entry) + "\n")

    n_hallu = sum(1 for e in benchmark if e["hallucinated"])
    type_counts = {}
    for e in benchmark:
        if e["hallu_type"]:
            type_counts[e["hallu_type"]] = type_counts.get(e["hallu_type"], 0) + 1

    print(f"[done] {len(benchmark)} samples → {out_path}")
    print(f"  hallucinated : {n_hallu}")
    print(f"  clean        : {len(benchmark) - n_hallu}")
    print(f"  type breakdown: {type_counts}")
    print()
    print("  [reminder] Small attribute/relational subsets (see type breakdown "
          "above) — report 95% CIs per Table 2, and see module docstring for "
          "the fixed-vocabulary limitation that should be stated in the paper.")

# ── Note on SWAP_MAP duplication ─────────────────────────────
# SWAP_MAP is intentionally duplicated (not imported from a shared module)
# between this file and build_benchmark_multi.py so that either script
# remains fully runnable standalone. If you ever change the injection
# vocabulary, you MUST update BOTH copies, or the legacy single-source
# benchmark and the multi-source benchmark will silently use different
# hallucination definitions — exactly the kind of quiet drift this whole
# review round was about. Consider factoring both into a shared
# `hallucination_vocab.py` in a future version if this becomes error-prone.