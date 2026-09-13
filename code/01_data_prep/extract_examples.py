#!/usr/bin/env python3
"""
01_data_prep/extract_examples.py — v2 (post-review)

Addresses Reviewer CTbs ("provide representative examples for each
hallucination type") and Reviewer 76GN ("no concrete example of an
image-claim pair... makes it difficult to assess granularity/quality").

CHANGE vs v1: this script is now called automatically by run_pipeline.sh
as part of Stage 1 (Data Preparation). Previously it existed but was never
invoked by the documented pipeline, so the paper's requested examples were
never actually produced unless someone remembered to run this file by
hand — that gap is now closed.

Run AFTER build_benchmark_multi.py (or build_benchmark.py for the legacy
single-source path — this script auto-detects which benchmark file
exists, preferring the multi-source one). Selects one clean example and
one example per hallucination type (object/attribute/relational), copies
the corresponding image into results/v1/examples/, and writes a markdown
file with the original vs. injected text spans highlighted, ready to
paste into the paper as a figure/table.

Usage:
  python extract_examples.py                  # one example per type, first match
  python extract_examples.py --seed 7          # different random example per type
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import yaml

ROOT     = Path(__file__).resolve().parents[3]
CFG_PATH = ROOT / "code/v1/config.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

PROCESSED = ROOT / CFG["paths"]["data_processed"]
EXAMPLES_DIR = ROOT / "results/v1/examples"
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

HALLU_TYPES = ["object", "attribute", "relational"]


def main(seed: int):
    # Prefer the multi-source benchmark (now the default), fall back to
    # the legacy single-source file if that's all that's been built.
    multi_path = PROCESSED / "benchmark_multi_v1.jsonl"
    single_path = PROCESSED / "benchmark_v1.jsonl"
    bench_path = multi_path if multi_path.exists() else single_path
    print(f"[source] Using {bench_path.name}")
    if not bench_path.exists():
        print(f"[error] Neither {multi_path.name} nor {single_path.name} found — "
              f"run build_benchmark_multi.py (default) or build_benchmark.py "
              f"(legacy) first.")
        return

    with open(bench_path) as f:
        samples = [json.loads(l) for l in f]

    rng = random.Random(seed)

    selected = {}
    clean_samples = [s for s in samples if not s["hallucinated"]]
    if clean_samples:
        selected["clean"] = rng.choice(clean_samples)

    for htype in HALLU_TYPES:
        matches = [s for s in samples if s["hallu_type"] == htype]
        if matches:
            selected[htype] = rng.choice(matches)
        else:
            print(f"[warn] no samples found for type '{htype}'")

    md_lines = [
        "# SAFER Benchmark — Representative Examples",
        "",
        "Auto-extracted for paper Section 3 (Dataset Construction), "
        "addressing Reviewer CTbs and Reviewer 76GN's request for concrete "
        "image-claim examples.",
        "",
    ]

    for key, s in selected.items():
        img_src = Path(s["img_path"])
        img_dst = EXAMPLES_DIR / f"example_{key}{img_src.suffix}"
        if img_src.exists():
            shutil.copy(img_src, img_dst)
            img_note = f"![{key} example]({img_dst.relative_to(ROOT)})"
        else:
            img_note = f"(image not found at {img_src})"

        md_lines.append(f"## {key.capitalize()} example (id={s['id']}, source={s.get('source', 'openi')})")
        md_lines.append("")
        md_lines.append(img_note)
        md_lines.append("")
        md_lines.append(f"**Original (faithful) report text:**")
        md_lines.append(f"> {s['original_report']}")
        md_lines.append("")
        if s["hallucinated"]:
            md_lines.append(f"**Injected claim shown to model:**")
            md_lines.append(f"> {s['report']}")
            md_lines.append("")
            md_lines.append(
                f"**Perturbation:** `{s['original_span']}` → `{s['injected_span']}` "
                f"(type: {s['hallu_type']})"
            )
        else:
            md_lines.append("**Claim shown to model:** identical to original "
                             "(clean/faithful sample, no perturbation applied).")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    out_md = EXAMPLES_DIR / "examples.md"
    with open(out_md, "w") as f:
        f.write("\n".join(md_lines))

    print(f"[done] {len(selected)} examples → {out_md}")
    print(f"  Images copied to → {EXAMPLES_DIR}")
    print("  [reminder] Review these examples for clinical sensibility before "
          "using in the paper — this script picks randomly by seed, it does "
          "not check for a particularly illustrative or clean-looking case.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(seed=args.seed)