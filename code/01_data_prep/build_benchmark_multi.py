"""
01_data_prep/build_benchmark_multi.py — v2 (post-review, config-driven)
Multi-dataset benchmark builder: OpenI (IU-Xray, full) + ReXGradient-160K.

THIS IS NOW THE DEFAULT BUILD SCRIPT called by run_pipeline.sh. The old
single-source build_benchmark.py (500 OpenI-only samples) is kept
unchanged and available for anyone who explicitly wants to reproduce the
original submission's exact numbers — see that file's docstring.

FINAL DATASET COMBINATION for this benchmark: OpenI + ReXGradient-160K.
A third dataset (PadChest) was evaluated and explicitly ruled out — its
publicly released report text is stemmed and tokenized (e.g. "compar con
estudi previ" instead of natural Spanish), not just a translation problem.
This is a genuine text-quality issue, not an access-friction one: injecting
hallucinations into already-mutilated, non-fluent stemmed text would
produce claims that don't read as plausible real reports, directly
undermining the same "does this look like a real report" requirement
reviewers already flagged. PadChest is not used in this benchmark for this
reason, not because of its 54-zip/1TB distribution size (which was a
secondary concern).

CHANGE vs v1: dataset selection (sources / n_total / injection_rate /
rexgradient_target_n) is now read from config.yaml's `dataset` block
instead of being hardcoded module-level constants or argparse-only
defaults. This closes a config/code-drift gap: previously config.yaml's
`dataset.n_samples: 500` had nothing to do with this script at all (only
the OLD single-source build_benchmark.py read it), which meant the config
file silently misrepresented what the "current" benchmark actually looks
like. CLI flags still exist and OVERRIDE config when explicitly passed,
so ad-hoc runs remain easy, but the config file is now the single source
of truth for the paper's reported dataset numbers when no override is given.

DESIGN — pluggable per-dataset adapters:
  Each dataset gets its own `load_<name>()` function returning a list of
  dicts with a COMMON schema:
    { "id": str (globally unique, prefixed by source),
      "source": str,
      "img_path": str,
      "report": str,
      "view_position": str }   # "frontal" | "lateral" | "unknown"

  This means adding a future dataset with genuinely natural report text
  (e.g. CheXpert Plus once image access clears, MIMIC-CXR once PhysioNet
  credentialing clears) is just one more load_<name>() function plus one
  line in DATASET_LOADERS below — the injection/inference/metrics code
  downstream never changes. PadChest is deliberately NOT included as a
  commented-out adapter here, to avoid the temptation of a quick re-add
  without re-solving its text-quality problem first.

ACCESS NOTES (confirmed via direct inspection, not assumed):
  - OpenI: already have, no gating.
  - ReXGradient-160K: requires accepting the license agreement on
    Hugging Face (self-serve, not institutional review, but not
    instantaneous either — do this first if you haven't:
    https://huggingface.co/datasets/rajpurkarlab/ReXGradient-160K).
    CONFIRMED STRUCTURE: report/metadata fields (Findings, Impression,
    StudyInstanceUid, etc.) plus per-image ImageViewPosition are in
    metadata/train_metadata_view_position.json (a dict keyed by composite
    study id). Images themselves are packed into 10 independently-valid
    zstd-compressed tar archives (deid_png.part00..part09, ~15.5GB each,
    ~150GB total) with NO random-access index — extraction requires
    sequentially streaming through each part. This script downloads the
    (small) metadata JSON, selects target_n frontal (PA/AP) studies from
    it WITHOUT touching the image archives, then streams each of the 10
    parts exactly once, extracting only the selected file paths and
    discarding the rest — this avoids storing the full ~150GB corpus on
    disk, but does NOT avoid reading ~150GB of compressed data over the
    network at least once, since the archive format has no way to skip
    directly to a specific file. Requires `zstd` and `tar` CLI tools
    (`sudo apt install zstd` if not already present) in addition to
    `pip install huggingface_hub --break-system-packages`.
  - This script assumes `huggingface-cli login` has already been run with
    a token that has accepted the ReXGradient-160K gated-dataset terms.

REVIEWER-RELEVANT NOTE ON SCALE (Reviewer 8sZM: "only 500 samples... too
small to support broad claims"): combining full OpenI (~3,955 report-image
pairs after frontal-only filtering) with a sampled subset of ReXGradient-160K
(default: up to 25,000 studies per config.yaml, frontal-view, English-report)
gives a combined pool in the tens of thousands, letting per-type
hallucination subsets (object/attribute/relational) scale well beyond the
n=25/n=32 the original reviewers flagged as too small for confident
per-type conclusions.
"""

import json
import random
import re
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

IMAGES_DIR  = RAW / "images" / "images_normalized"
REPORTS_CSV = RAW / "indiana_reports.csv"
PROJ_CSV    = RAW / "indiana_projections.csv"

REXGRADIENT_CACHE_DIR = RAW / "rexgradient_cache"
REXGRADIENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DATASET_CFG = CFG["dataset"]
SEED = DATASET_CFG["random_seed"]
random.seed(SEED)

# Config-driven defaults (CLI flags below can still override any of these).
DEFAULT_SOURCES         = DATASET_CFG["sources"]
DEFAULT_N_TOTAL         = DATASET_CFG["n_total"]          # None = use everything
DEFAULT_INJECTION_RATE  = DATASET_CFG["injection_rate"]
DEFAULT_REXGRADIENT_N   = DATASET_CFG["rexgradient_target_n"]
# NEW: per-source cap (e.g. 4500 from OpenI AND 4500 from ReXGradient = 9000
# total). .get() with a None default so older configs without this key
# still work unchanged (falls back to legacy pool-then-truncate via n_total).
DEFAULT_PER_SOURCE_CAP  = DATASET_CFG.get("per_source_cap", None)


# ── Hallucination vocabulary (unchanged from single-dataset version) ────────
SWAP_MAP = {
    "object": {
        # original terms
        "lung": "liver", "heart": "kidney", "pleura": "diaphragm",
        "aorta": "trachea", "rib": "clavicle", "spine": "sternum",
        # new — high hit rate in failed reports
        "effusion": "consolidation",   # 71.2% of failures
        "consolidation": "effusion",   # 37.6% of failures (reverse)
        "pneumothorax": "effusion",    # 63.6% of failures
        "edema": "consolidation",      # 11.5% of failures
        "atelectasis": "consolidation",# 6.1% of failures
        "nodule": "mass",              # 3.6%
        "mass": "nodule",              # 3.0%
    },
    "attribute": {
        # original terms
        "large": "small", "small": "large",
        "increased": "decreased", "decreased": "increased",
        "bilateral": "unilateral", "unilateral": "bilateral",
        "mild": "severe", "severe": "mild",
        # new — high hit rate in failed reports
        "normal": "abnormal",          # 77.7% of failures — biggest win
        "acute": "chronic",            # 66.2% of failures
        "chronic": "acute",            # covers reverse
        "clear": "opacified",          # 48.7% of failures
        "focal": "diffuse",            # 43.9% of failures
        "diffuse": "focal",            # reverse
        "stable": "new",               # 7.0% of failures
        "unremarkable": "abnormal",    # 18.7% of failures
    },
    "relational": {
        # original terms
        "left": "right", "right": "left",
        "upper": "lower", "lower": "upper",
        "anterior": "posterior", "posterior": "anterior",
        # new — high hit rate in failed reports
        "pleural": "parenchymal",      # 61.6% of failures
        "hilar": "pleural",            # 17.0% of failures
        "mediastinal": "pleural",      # covers "mediastin" hits (67.9%)
        "airspace": "interstitial",    # 11.7% of failures
        "interstitial": "airspace",    # reverse
    },
}

def inject(text: str, htype: str) -> tuple:
    for orig, fake in SWAP_MAP[htype].items():
        pattern = re.compile(re.escape(orig), re.IGNORECASE)
        if pattern.search(text):
            new_text = pattern.sub(fake, text, count=1)
            return new_text, orig, fake
    return text, "", ""


# ── Adapter: OpenI (IU-Xray) — FULL dataset, not just 500 ──────────────────
def load_openi_full() -> list[dict]:
    """
    Loads ALL valid frontal image-report pairs from OpenI (~3,955 after
    filtering), not a pre-truncated 500-sample subset. The old
    build_benchmark.py truncated to config's legacy_single_source
    .n_samples=500 inside build(); this loader returns everything usable.

    IMPORTANT: this loader does NOT itself cap the count. Per-source caps
    (e.g. "4500 from OpenI, 4500 from ReXGradient") are applied centrally
    in build_multi_source() below via `per_source_cap`, AFTER loading —
    OpenI's true yield (~3,955) is naturally below a 4500 cap and that's
    accepted as-is (not an error, not padded from another source). This
    keeps the per-dataset loaders as pure "give me everything valid"
    functions, easy to audit and reuse, while the actual balancing logic
    lives in exactly one place.
    """
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
            "id":            f"openi_{row['uid']}",
            "source":        "openi",
            "img_path":      row["img_path"],
            "report":        report,
            "view_position": "frontal",
        })
    return records


# ── Adapter: ReXGradient-160K ────────────────────────────────────────────────
def load_rexgradient(target_n: int = None) -> list[dict]:
    """
    REAL implementation, based on the confirmed actual dataset structure
    (verified via direct inspection — see project diagnostics):

      - metadata/train_metadata_view_position.json is a dict keyed by the
        composite study id "p<PatientID>_a<AccessionNumber>_s<StudyUID>".
        Each value has ImagePath (list), ImageViewPosition (list, aligned
        by index with ImagePath), Findings, Impression, etc. A single
        study can have multiple images (e.g. one 'PA' + one 'LATERAL');
        we pick the FIRST image whose ImageViewPosition is 'PA' or 'AP'.
      - Images themselves live inside 10 independently-valid zstd-
        compressed tar archives (deid_png.part00 .. part09), each
        ~15.5GB, ~150GB total. Each part decompresses to paths rooted at
        "deid_png/...". There is NO random-access index inside a part —
        extraction is sequential — but each part IS independently valid
        (does not require concatenation with other parts first).

    STRATEGY: rather than downloading and fully extracting all ~150GB to
    find ~target_n frontal images, we:
      1. Download the (small) view-position JSON — already local after
         the first call, or fetched once via hf_hub_download.
      2. Select target_n studies with a valid PA/AP image from the JSON,
         WITHOUT touching the image archives yet.
      3. For each of the 10 archive parts, stream-decompress
         (`zstd -dc partNN | tar -x --wildcards -T <selected_paths_file>`)
         extracting ONLY the specific selected file paths that fall in
         that part, discarding everything else in the stream.
      4. Because we don't know in advance which part contains which
         study's image (paths aren't part-indexed in the metadata), we
         must scan every part once per run. This still means reading
         ~150GB of compressed data through zstd sequentially, but we only
         WRITE the files we actually selected, not the full 273,004-image
         corpus — a large disk-space saving even though download/
         decompression time is not avoidable given the archive format has
         no random-access index.

    This is slower than a naive "just grab what we need" hope, but it is
    the real, verified-correct approach given the archive format — not a
    guess. Expect this step to take substantial time/bandwidth on first
    run; results are cached to REXGRADIENT_CACHE_DIR so re-runs with the
    same target_n reuse already-extracted images.

    target_n: if None, uses DEFAULT_REXGRADIENT_N from config.yaml
    (dataset.rexgradient_target_n). Explicit argument still overrides.
    """
    import subprocess
    from huggingface_hub import hf_hub_download

    if target_n is None:
        target_n = DEFAULT_REXGRADIENT_N

    REPO = "rajpurkarlab/ReXGradient-160K"
    N_PARTS = 10

    print("[rexgradient] Step 1/3: loading view-position + report metadata ...")
    vp_json_path = hf_hub_download(
        repo_id=REPO, repo_type="dataset",
        filename="metadata/train_metadata_view_position.json",
    )
    with open(vp_json_path) as f:
        vp_data = json.load(f)
    print(f"  → {len(vp_data)} studies in metadata")

    # ── Step 2: select up to target_n studies with a usable frontal image ──
    print(f"[rexgradient] Step 2/3: selecting up to {target_n} frontal studies ...")
    selected = []  # list of (study_key, chosen_image_path, report_text)
    n_skipped_no_frontal = 0
    n_skipped_empty_report = 0

    study_keys = list(vp_data.keys())
    random.Random(SEED).shuffle(study_keys)  # avoid always picking alphabetically-first patients

    for key in study_keys:
        if len(selected) >= target_n:
            break
        entry = vp_data[key]
        image_paths = entry.get("ImagePath", [])
        view_positions = entry.get("ImageViewPosition", [])

        frontal_idx = None
        for idx, vp in enumerate(view_positions):
            if str(vp).upper() in ("PA", "AP"):
                frontal_idx = idx
                break
        if frontal_idx is None or frontal_idx >= len(image_paths):
            n_skipped_no_frontal += 1
            continue

        findings   = str(entry.get("Findings", "") or "").strip()
        impression = str(entry.get("Impression", "") or "").strip()
        report = f"{findings} {impression}".strip()
        if len(report) < 20:
            n_skipped_empty_report += 1
            continue

        # ImagePath is relative, like "../deid_png/PatientID/.../instances/X.png"
        # Normalize to the "deid_png/..." form matching actual archive paths.
        raw_path = image_paths[frontal_idx]
        archive_member_path = raw_path.split("../", 1)[-1] if "../" in raw_path else raw_path
        archive_member_path = archive_member_path.lstrip("/")
        if not archive_member_path.startswith("deid_png/"):
            archive_member_path = "deid_png/" + archive_member_path.split("deid_png/", 1)[-1] \
                if "deid_png/" in archive_member_path else archive_member_path

        selected.append({
            "study_key": key,
            "archive_member_path": archive_member_path,
            "report": report,
        })

    print(f"  → selected {len(selected)} studies "
          f"(skipped {n_skipped_no_frontal} with no PA/AP image, "
          f"{n_skipped_empty_report} with empty report)")

    if not selected:
        print("  [error] No usable studies selected — check metadata format "
              "hasn't changed, or lower target_n.")
        return []

    # ── Step 3: stream-extract only the selected image files from the 10 parts ──
    print(f"[rexgradient] Step 3/3: extracting {len(selected)} images from "
          f"{N_PARTS} archive parts (this reads ~150GB of compressed data "
          f"sequentially but only WRITES the selected files) ...")

    wanted_paths = {s["archive_member_path"] for s in selected}
    already_have = set()
    for s in selected:
        out_path = REXGRADIENT_CACHE_DIR / Path(s["archive_member_path"]).name
        if out_path.exists():
            already_have.add(s["archive_member_path"])
    still_needed = wanted_paths - already_have
    print(f"  → {len(already_have)} already cached, {len(still_needed)} to extract")

    if still_needed:
        for part_idx in range(N_PARTS):
            if not still_needed:
                break
            part_name = f"deid_png.part{part_idx:02d}"
            print(f"  [part {part_idx+1}/{N_PARTS}] downloading + scanning {part_name} ...")
            part_path = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=part_name)

            # FIX #1 (symlink): hf_hub_download returns a path that is a
            # SYMLINK into HF's content-addressed cache store. `zstd`
            # refuses to read symlinks by default ("is a symbolic link,
            # ignoring"), silently producing an empty stream. Resolve to
            # the real underlying path once, reused throughout this loop.
            real_part_path = str(Path(part_path).resolve())
            if not Path(real_part_path).exists():
                print(f"    [error] resolved path {real_part_path} does not "
                      f"exist — symlink may be broken, skipping this part")
                continue

            # FIX #3: HF's local cache does not automatically detect or
            # repair partial/truncated downloads left behind by an
            # interrupted previous run (e.g. Ctrl+C mid-download). A
            # truncated .zst file causes `zstd -dc` to fail with
            # "Read error (39): premature end" partway through, silently
            # losing every remaining file in that part. We verify each
            # cached part decompresses cleanly BEFORE attempting selective
            # extraction from it, and force a clean re-download if not.
            integrity_check = subprocess.run(
                f"zstd -t '{real_part_path}'",
                shell=True, capture_output=True, text=True,
            )
            if integrity_check.returncode != 0:
                print(f"    [warn] {part_name} failed zstd integrity check "
                      f"({integrity_check.stderr.strip()}) — likely a "
                      f"truncated download from an earlier interrupted run. "
                      f"Removing cached blob and re-downloading...")
                try:
                    # Remove the actual blob (not just the symlink) so
                    # hf_hub_download is forced to fetch a fresh copy.
                    Path(real_part_path).unlink()
                except Exception as e:
                    print(f"    [error] could not remove corrupted blob: {e}")
                part_path = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=part_name)
                real_part_path = str(Path(part_path).resolve())
                integrity_check2 = subprocess.run(
                    f"zstd -t '{real_part_path}'",
                    shell=True, capture_output=True, text=True,
                )
                if integrity_check2.returncode != 0:
                    print(f"    [error] {part_name} still fails integrity "
                          f"check after re-download — skipping this part. "
                          f"({integrity_check2.stderr.strip()})")
                    continue
                print(f"    [ok] {part_name} re-downloaded and verified clean.")
            # FIX #2 (the big one, confirmed via isolated testing): GNU
            # tar's `--files-from=FILE` (-T) for EXTRACTION prints matched
            # member names in verbose mode but does NOT actually write them
            # to disk in this tar version/configuration — it is a
            # documented flag but effectively a no-op for -x in practice
            # here. The reliable alternative, verified end-to-end through
            # the exact zstd-pipe + --transform combination used in
            # production, is passing wanted paths as POSITIONAL arguments
            # to `tar -x`, not via -T. Since positional args are subject to
            # OS ARG_MAX (~2MB on Linux) and 25,000 realistic archive paths
            # can exceed that, we BATCH the positional-arg extraction calls.
            wanted_this_part = list(still_needed)
            batch_size = 500  # conservative: 500 paths * ~150 chars << ARG_MAX
            n_batches = (len(wanted_this_part) + batch_size - 1) // batch_size

            for batch_idx in range(n_batches):
                batch = wanted_this_part[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                if not batch:
                    continue
                quoted_paths = " ".join(f"'{p}'" for p in batch)
                extract_cmd = (
                    f"zstd -f -dc '{real_part_path}' | "
                    f"tar --extract --directory='{REXGRADIENT_CACHE_DIR}' "
                    f"--transform='s|.*/||' --ignore-failed-read "
                    f"{quoted_paths} "
                    f"2>>'{REXGRADIENT_CACHE_DIR}/_extract_stderr_{part_idx}.log'"
                )
                result = subprocess.run(extract_cmd, shell=True)
                if result.returncode not in (0, 1, 2):
                    print(f"    [warn] batch {batch_idx+1}/{n_batches} of "
                          f"{part_name} returned code {result.returncode}")

            # Recompute what's still missing after this part
            still_needed = {
                p for p in still_needed
                if not (REXGRADIENT_CACHE_DIR / Path(p).name).exists()
            }
            print(f"    → {len(still_needed)} still needed after this part")

            # Delete the downloaded part after use to save disk, since HF's
            # cache would otherwise keep all 10 parts (~150GB) on disk.
            try:
                Path(part_path).unlink()
            except Exception:
                pass  # leave it if it's shared/symlinked elsewhere; not fatal

    # ── Assemble final records ──
    records = []
    for s in selected:
        img_path = REXGRADIENT_CACHE_DIR / Path(s["archive_member_path"]).name
        if not img_path.exists():
            continue  # extraction didn't find this one in any part; skip rather than fake it
        records.append({
            "id":            f"rexgradient_{s['study_key']}",
            "source":        "rexgradient",
            "img_path":      str(img_path),
            "report":        s["report"],
            "view_position": "frontal",
        })

    print(f"[rexgradient] Done: {len(records)}/{len(selected)} selected studies "
          f"successfully extracted with a usable local image.")
    if len(records) < len(selected) * 0.9:
        print(f"  [warn] more than 10% of selected studies could not be "
              f"extracted — check _extract_stderr_*.log files in "
              f"{REXGRADIENT_CACHE_DIR} for issues before trusting this count.")
    return records

# ── Adapter: MIMIC-CXR (Kaggle mirror — simhadrisadaram) ─────────────────────
def load_mimiccxr() -> list[dict]:
    """
    Loads frontal PA-view image-report pairs from the Kaggle MIMIC-CXR mirror.
    Layout under data/v1/raw/mimiccxr/:
      mimic_cxr_aug_train.csv        ← columns: subject_id, PA, text, ...
      mimic_cxr_aug_validate.csv     ← same schema
      official_data_iccv_final/
        files/p10/p10000032/s50414267/<dicom>.jpg  ← image paths in PA column
                                                      are relative to here

    The PA column is a stringified Python list of paths like:
      "['files/p10/p10000032/s53911762/68b5...jpg', ...]"
    We take the FIRST entry per row (one image per sample).
    The text column is also a stringified list (one report per study);
    we take the FIRST entry.
    Both train and validate CSVs are loaded (no label leakage risk here —
    we're only using image+report content, not split labels).
    """
    import ast

    MIMIC_DIR   = RAW / "mimiccxr"
    IMAGES_ROOT = MIMIC_DIR / "official_data_iccv_final"
    CSV_PATHS   = [
        MIMIC_DIR / "mimic_cxr_aug_train.csv",
        MIMIC_DIR / "mimic_cxr_aug_validate.csv",
    ]

    dfs = []
    for p in CSV_PATHS:
        if p.exists():
            dfs.append(pd.read_csv(p))
        else:
            print(f"  [mimiccxr] warning: {p.name} not found, skipping")
    if not dfs:
        raise FileNotFoundError(
            f"No MIMIC-CXR CSVs found under {MIMIC_DIR}. "
            "Expected mimic_cxr_aug_train.csv and/or mimic_cxr_aug_validate.csv."
        )
    meta = pd.concat(dfs, ignore_index=True)

    records = []
    skipped_no_img = 0
    skipped_no_report = 0
    skipped_img_missing = 0

    for _, row in meta.iterrows():
        # PA column: stringified list of PA-view paths; take first entry
        pa_raw = str(row.get("PA", "") or "").strip()
        if not pa_raw or pa_raw in ("nan", "[]", "['']"):
            skipped_no_img += 1
            continue
        try:
            pa_list = ast.literal_eval(pa_raw)
        except Exception:
            skipped_no_img += 1
            continue
        if not pa_list:
            skipped_no_img += 1
            continue
        rel_path = pa_list[0]  # e.g. "files/p10/p10000032/s53911762/68b5....jpg"

        img_path = IMAGES_ROOT / rel_path
        if not img_path.exists():
            skipped_img_missing += 1
            continue

        # text column: stringified list of report strings; take first entry
        text_raw = str(row.get("text", "") or "").strip()
        if not text_raw or text_raw in ("nan", "[]", "['']"):
            skipped_no_report += 1
            continue
        try:
            text_list = ast.literal_eval(text_raw)
        except Exception:
            skipped_no_report += 1
            continue
        report = text_list[0].strip() if text_list else ""
        if len(report) < 20:
            skipped_no_report += 1
            continue

        subject_id = str(row.get("subject_id", "unknown"))
        # Extract study_id from the path: files/p10/p10000032/s50414267/...
        path_parts = Path(rel_path).parts  # ('files','p10','p10000032','s50414267','<dicom>.jpg')
        study_id = path_parts[3] if len(path_parts) >= 5 else Path(rel_path).parent.name

        records.append({
            "id":            f"mimiccxr_{subject_id}_{study_id}",
            "source":        "mimiccxr",
            "img_path":      str(img_path),
            "report":        report,
            "view_position": "PA",
        })

    print(f"[mimiccxr] {len(records)} valid PA-view records loaded "
          f"(skipped: {skipped_no_img} no PA path, "
          f"{skipped_img_missing} image file missing, "
          f"{skipped_no_report} empty report)")
    return records

DATASET_LOADERS = {
    "openi":    load_openi_full,
    "mimiccxr": load_mimiccxr,
    # Future adapters can be added here for datasets with genuinely
    # natural (non-stemmed) report text, e.g.:
    # "chexpert_plus": load_chexpert_plus, # needs CheXpert image access
    # "mimic_cxr":    load_mimic_cxr,      # needs PhysioNet credentialing
}


# ── Combine + build benchmark ────────────────────────────────────────────────
def build_multi_source(sources: list[str], n_total, injection_rate: float,
                        per_source_cap: int = None) -> list[dict]:
    """
    per_source_cap (NEW): if set, each source is independently capped at
    this many records BEFORE combining, rather than loading everything
    from every source, pooling it all together, shuffling, and truncating
    the COMBINED pool to n_total. That old pool-then-truncate approach has
    a real bug for uneven sources: if OpenI naturally yields ~3,955 records
    and ReXGradient yields 25,000+, a global shuffle+truncate to n_total
    would let whichever source happens to be larger dominate the final
    mix — you'd never reliably get "4500 from OpenI AND 4500 from
    ReXGradient", you'd get "however many OpenI has, plus ReXGradient
    filling in the rest." per_source_cap fixes this by capping each
    source's contribution independently, so a request like "4500 from
    each of 2 sources" is actually honored per-source, with each source's
    true (possibly smaller) yield reported transparently rather than
    silently topped up by another source.

    If a source's loader returns FEWER records than per_source_cap (e.g.
    OpenI's ~3,955 vs a 4500 cap), that source's full yield is used as-is
    — this is accepted, not treated as an error, and is reported clearly
    in per_source_counts / per_source_capped_counts so the discrepancy is
    visible rather than hidden.

    n_total (legacy behavior, still supported): if per_source_cap is NOT
    set, falls back to the original pool-then-truncate-to-n_total
    behavior for backward compatibility with configs that only set
    dataset.n_total.
    """
    all_records = []
    per_source_counts = {}          # raw yield per source, before any capping
    per_source_capped_counts = {}   # actual count used per source, after capping
    for src in sources:
        if src not in DATASET_LOADERS:
            print(f"[skip] unknown source '{src}' — not in DATASET_LOADERS")
            continue
        print(f"\n[load] {src} ...")
        recs = DATASET_LOADERS[src]()
        per_source_counts[src] = len(recs)

        if per_source_cap is not None:
            random.shuffle(recs)
            if len(recs) > per_source_cap:
                recs = recs[:per_source_cap]
                print(f"  → {src}: capped to {per_source_cap} "
                      f"(had {per_source_counts[src]} available)")
            else:
                print(f"  → {src}: using all {len(recs)} available "
                      f"(BELOW the {per_source_cap} cap — this source "
                      f"simply doesn't have more; not an error, not "
                      f"padded from another source)")
            per_source_capped_counts[src] = len(recs)

        all_records.extend(recs)

    print(f"\n[combine] {len(all_records)} total records "
          f"across {len(sources)} sources")
    print(f"  raw available per source   : {per_source_counts}")
    if per_source_cap is not None:
        print(f"  used per source (capped)   : {per_source_capped_counts}")
        print(f"  [reminder] Report per_source_capped_counts above verbatim "
              f"in the paper's Data Construction section — if any source "
              f"came in under the cap, say so explicitly rather than "
              f"implying an even split that didn't actually happen.")

    random.shuffle(all_records)

    if per_source_cap is not None:
        # Per-source caps already applied above; do NOT additionally
        # truncate the combined pool by n_total, or we'd reintroduce the
        # exact "larger source dominates" bug this fix is meant to avoid.
        if n_total is not None:
            print(f"  [note] per_source_cap is set — ignoring n_total="
                  f"{n_total!r} for the combine step (n_total only applies "
                  f"in the legacy pool-then-truncate mode, i.e. when "
                  f"per_source_cap is NOT set).")
    elif n_total is not None and n_total < len(all_records):
        all_records = all_records[:n_total]
        print(f"[sample] Downsampled to n_total={n_total} "
              f"(legacy pool-then-truncate mode — no per-source guarantee)")
    else:
        print(f"[sample] Using all {len(all_records)} loaded records "
              f"(n_total={n_total!r} in config means 'no cap')")

    n_hallu = int(len(all_records) * injection_rate)
    htypes  = (["object", "attribute", "relational"] * n_hallu)[:n_hallu]
    random.shuffle(htypes)

    benchmark = []
    for i, rec in enumerate(all_records):
        entry = {
            "id":              rec["id"],
            "source":          rec["source"],
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


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=None,
                        help="Which dataset adapters to combine "
                             "(default: config.yaml dataset.sources)")
    parser.add_argument("--n_total", type=int, default=None,
                        help="Cap on total combined benchmark size "
                             "(default: config.yaml dataset.n_total, "
                             "which is null/None = use everything loaded)")
    parser.add_argument("--injection_rate", type=float, default=None,
                        help="default: config.yaml dataset.injection_rate")
    parser.add_argument("--rexgradient_n", type=int, default=None,
                        help="How many ReXGradient studies to pull "
                             "(default: config.yaml dataset.rexgradient_target_n)")
    parser.add_argument("--per_source_cap", type=int, default=None,
                        help="Cap EACH source at this many records "
                             "independently (e.g. 4500 = 4500 from OpenI "
                             "AND 4500 from ReXGradient, not a combined-"
                             "pool truncation). Overrides n_total's "
                             "truncation behavior when set. Default: "
                             "config.yaml dataset.per_source_cap (null "
                             "= legacy pool-then-truncate-by-n_total mode).")
    args = parser.parse_args()

    # CLI flags override config when explicitly passed; otherwise config
    # is the single source of truth (closes the config/code drift gap).
    sources         = args.sources if args.sources is not None else DEFAULT_SOURCES
    n_total         = args.n_total if args.n_total is not None else DEFAULT_N_TOTAL
    injection_rate  = args.injection_rate if args.injection_rate is not None else DEFAULT_INJECTION_RATE
    per_source_cap  = args.per_source_cap if args.per_source_cap is not None else DEFAULT_PER_SOURCE_CAP

    # If a per-source cap is active, there's no reason to make
    # load_rexgradient() select (and then scan all 10 archive parts for)
    # more studies than the cap actually needs — e.g. asking for 25,000
    # candidate studies just to keep 4,500 of them wastes a full pass over
    # the ~150GB of compressed archives for nothing. Default rexgradient_n
    # to the cap itself unless the user explicitly passed --rexgradient_n.
    if args.rexgradient_n is not None:
        rexgradient_n = args.rexgradient_n
    elif per_source_cap is not None:
        rexgradient_n = per_source_cap
    else:
        rexgradient_n = DEFAULT_REXGRADIENT_N

# rexgradient no longer in DATASET_LOADERS — mimiccxr is the second source

    print(f"[config] sources={sources}  n_total={n_total!r}  "
          f"per_source_cap={per_source_cap!r}  "
          f"injection_rate={injection_rate}  rexgradient_n={rexgradient_n}")

    benchmark = build_multi_source(
        sources=sources,
        n_total=n_total,
        injection_rate=injection_rate,
        per_source_cap=per_source_cap,
    )

    out_path = PROCESSED / "benchmark_multi_v1.jsonl"
    with open(out_path, "w") as f:
        for entry in benchmark:
            f.write(json.dumps(entry) + "\n")

    n_hallu = sum(1 for e in benchmark if e["hallucinated"])
    type_counts, source_counts = {}, {}
    for e in benchmark:
        if e["hallu_type"]:
            type_counts[e["hallu_type"]] = type_counts.get(e["hallu_type"], 0) + 1
        source_counts[e["source"]] = source_counts.get(e["source"], 0) + 1

    print(f"\n[done] {len(benchmark)} samples → {out_path}")
    print(f"  hallucinated : {n_hallu}")
    print(f"  clean        : {len(benchmark) - n_hallu}")
    print(f"  type breakdown   : {type_counts}")
    print(f"  source breakdown : {source_counts}")
    print()
    print("  [reminder] Report the exact source_counts and type_counts above "
          "verbatim in the paper's Data Construction section — do not round "
          "or approximate these, reviewers will check consistency against "
          "any table you report.")