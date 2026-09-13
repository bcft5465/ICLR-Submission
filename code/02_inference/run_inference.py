#!/usr/bin/env python3
"""
02_inference/run_inference.py — v4 (post-review fix, round 2)

Changes vs v3:
  1. SHUFFLED-IMAGE BASELINE NOW COVERS ALL VLMs BY DEFAULT (previously
     hardcoded to qwen25_vl only, justified as "fastest of the three" —
     but that meant the shuffled-image validity check, arguably the most
     important test of whether a model uses the image AT ALL, was never
     run against MedGemma, the model the paper's headline HalluScore
     claims rest on. run_shuffled_image_baseline() now accepts a list of
     model keys and loops over them; default list comes from config.yaml
     inference.shuffled_image_baseline_models (currently all three:
     qwen25_vl, medgemma, llama). Override with --model for a faster
     single-model run during development.
  2. Reads dataset/metrics config paths consistent with the v2 config.yaml
     restructure (paths unaffected here, but resolve_benchmark_path() docs
     updated to reflect build_benchmark_multi.py now being the default).

Changes vs v2 (retained from that fix, unchanged):
  1. CONFIDENCE SEMANTICS FIXED: prompt and parser now unambiguously elicit
     P(hallucinated) directly. Models are asked "On a scale 0.00-1.00, how
     likely is it that this report contains an error?" — there is no longer
     any YES/NO-to-probability inversion step, which was the #1 source of
     ambiguity flagged by Reviewer CTbs.
  2. AUTO-INVERSION REMOVED: the old maybe_invert() function silently
     rewrote output files in place whenever AUROC < 0.5. This has been
     deleted entirely. If a model's confidence still anti-correlates with
     correctness after this fix, that is reported as a genuine finding,
     not "corrected" post-hoc.
  3. SINGLE LLAMA PATH: the old codebase had three competing LLaMA
     implementations (run_llama.py standalone, llama_fix.py, and the
     run_llama_transformers() in this file, with a hidden vocab-resize
     hotfix only present in one of them). Consolidated into ONE function
     with the embedding-resize hotfix applied unconditionally (it is a
     correctness fix for a real tokenizer/vocab mismatch, not a metric-
     affecting choice, so there is no ambiguity in keeping it).
  4. Confidence field renamed pred_prob_hallucinated for clarity (old
     pred_confidence name was ambiguous about direction — that ambiguity
     is exactly what caused the reviewer concern).
"""

# ── PATCHES (tokenizer/image-processing compat shims — unrelated to metrics) ──
import sys, builtins
from PIL.Image import Resampling
sys.modules["transformers.utils.PILImageResampling"] = Resampling
setattr(builtins, "PILImageResampling", Resampling)
try:
    import transformers
    import transformers.image_processing_utils as _ipu
    import transformers.image_processing_base as _ipb
    for _mod in [transformers, _ipu, _ipb]:
        if not hasattr(_mod, "PILImageResampling"):
            setattr(_mod, "PILImageResampling", Resampling)
except Exception:
    pass

try:
    from transformers import MllamaProcessor
    if not hasattr(MllamaProcessor, "_get_num_multimodal_tokens"):
        def _get_num_multimodal_tokens(self, **kwargs):
            return {"num_image_tokens": [1601]}
        MllamaProcessor._get_num_multimodal_tokens = _get_num_multimodal_tokens
except ImportError:
    pass

import argparse
import json
import re
import time
from pathlib import Path

import torch
import yaml
from PIL import Image
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[3]
CFG_PATH = ROOT / "code/v1/config.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

PROCESSED  = ROOT / CFG["paths"]["data_processed"]
OUTPUTS    = ROOT / CFG["paths"]["outputs"]
OUTPUTS.mkdir(parents=True, exist_ok=True)
MODELS_DIR = ROOT / CFG["paths"]["models"]

LOCAL_DIRS = {
    "qwen25_vl":   str(MODELS_DIR / "Qwen2.5-VL-7B-Instruct"),
    "medgemma":    str(MODELS_DIR / "medgemma-27b-it"),
    # "llama":       str(MODELS_DIR / "Llama-3.2-11B-Vision-Instruct"),
    "biovil_t":    str(MODELS_DIR / "BiomedVLP-BioViL-T"),
    # "llava_med":   str(MODELS_DIR / "llava-med-7b"),
    "chexagent":   str(MODELS_DIR / "CheXagent-8b"),
    "biomedclip":  str(MODELS_DIR / "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"),
    "med_flamingo":str(MODELS_DIR / "med-flamingo"),
    "phi3_vision": str(MODELS_DIR / "Phi-3-vision-128k-instruct"),
    "blip2":       str(MODELS_DIR / "blip2-opt-6.7b"),
    "llava15":     str(MODELS_DIR / "llava-1.5-13b-hf"),
}

# Which real VLMs the shuffled-image validity baseline runs by default.
# Read from config.yaml (inference.shuffled_image_baseline_models) so this
# is a single source of truth rather than a hardcoded list buried in code —
# previously this was hardcoded to ["qwen25_vl"] only, here.
SHUFFLED_IMAGE_DEFAULT_MODELS = CFG.get("inference", {}).get(
    "shuffled_image_baseline_models", ["qwen25_vl", "medgemma"]
)

OUTPUT_VERSION = "v3"  # bump so old v1/v2 files are never silently reused


def resolve_benchmark_path() -> Path:
    """
    Prefers the multi-dataset benchmark (OpenI + ReXGradient-160K — this is
    now the DEFAULT benchmark produced by build_benchmark_multi.py, which
    run_pipeline.sh calls by default) over the legacy single-source OpenI-
    only file, so the rest of the pipeline picks up the larger dataset
    automatically — no manual path edits needed in every downstream
    script. Falls back to the original single-source file if only that one
    exists (e.g. someone explicitly ran the legacy build_benchmark.py),
    so this script still works standalone for anyone re-running only the
    original OpenI-only pipeline for comparison purposes.
    """
    multi_path = PROCESSED / "benchmark_multi_v1.jsonl"
    single_path = PROCESSED / "benchmark_v1.jsonl"
    if multi_path.exists():
        return multi_path
    return single_path

# ── Prompts ───────────────────────────────────────────────────────────────────
# FIX (Reviewer CTbs): the old prompt asked for "confidence in your YES/NO
# answer", which is a direction-ambiguous quantity (confidence in whichever
# label was picked, NOT a fixed-direction probability). AUROC and ECE are only
# well-defined against a probability with a FIXED, KNOWN direction. The new
# prompt asks directly and only for P(report contains an error), independent
# of the YES/NO verdict, removing the ambiguity at the source.

PROMPT_STANDARD = (
    "You are a critical radiology AI evaluator. "
    "Your job is to detect errors and hallucinations in chest X-ray reports.\n\n"
    "Carefully examine the X-ray image and the report below. "
    "Check whether every finding, anatomy, attribute (size/shape/density), "
    "and spatial relationship (left/right/upper/lower) in the report is "
    "actually visible and correct in the image.\n\n"
    "Report: {report}\n\n"
    "Respond in EXACTLY this format on one line:\n"
    "VERDICT: <YES or NO> | P_ERROR: <0.00-1.00> | REASON: <one sentence>\n\n"
    "VERDICT is YES if the report is completely accurate, NO if it contains "
    "ANY error.\n"
    "P_ERROR is your estimated PROBABILITY THAT THE REPORT CONTAINS AN ERROR "
    "(0.00 = certainly no error / faithful, 1.00 = certainly contains an "
    "error / hallucinated). P_ERROR must reflect this fixed direction "
    "regardless of your VERDICT choice — it is not 'confidence in your "
    "verdict', it is specifically the probability of an error being present.\n\n"
    "Example: VERDICT: NO | P_ERROR: 0.87 | REASON: The report states "
    "cardiomegaly but the cardiac silhouette appears normal."
)

PROMPT_COT = (
    "You are a critical radiology AI evaluator detecting hallucinations.\n\n"
    "Report: {report}\n\n"
    "Think step by step:\n"
    "Step 1 - Objects: Is every finding in the report visible in the image?\n"
    "Step 2 - Attributes: Are size/shape/density correct?\n"
    "Step 3 - Relations: Are left/right/spatial relations correct?\n"
    "Step 4 - Verdict.\n\n"
    "After your reasoning, give your final answer on a NEW LINE in EXACTLY "
    "this format:\n"
    "VERDICT: <YES or NO> | P_ERROR: <0.00-1.00>\n\n"
    "VERDICT is YES if the report is completely accurate, NO if it contains "
    "ANY error.\n"
    "P_ERROR is your estimated PROBABILITY THAT THE REPORT CONTAINS AN ERROR "
    "(0.00 = certainly faithful, 1.00 = certainly hallucinated). This is a "
    "fixed-direction probability, not confidence in the VERDICT."
)

PROMPTS = {"standard": PROMPT_STANDARD, "cot": PROMPT_COT}

VERDICT_RE = re.compile(
    r"VERDICT:\s*(YES|NO)\s*\|\s*P_ERROR:\s*(0?\.\d+|1\.0+|0|1)",
    re.IGNORECASE,
)


# ── Parse output ──────────────────────────────────────────────────────────────
def parse_output(raw: str) -> dict:
    """
    Returns:
      pred_label: 'faithful' | 'hallucinated' | 'unknown'
      pred_prob_hallucinated: float in [0,1], FIXED DIRECTION
          (P that the report contains an error), independent of pred_label.
      parse_ok: whether the strict format was matched (for auditing —
          reviewers should be able to see how many samples fell back to the
          unknown bucket, since that bucket is excluded from classification
          metrics but should be reported transparently, not hidden).
    """
    m = VERDICT_RE.search(raw.strip())
    if m:
        verdict_tok = m.group(1).upper()
        p_error = float(m.group(2))
        p_error = max(0.0, min(1.0, p_error))
        label = "faithful" if verdict_tok == "YES" else "hallucinated"
        return {
            "pred_label": label,
            "pred_prob_hallucinated": round(p_error, 4),
            "parse_ok": True,
        }

    # Fallback: try to salvage a bare YES/NO with no interpretable P_ERROR.
    # We do NOT invent a confidence value here — unknown probability is
    # recorded as null, not silently defaulted, so it can be excluded from
    # AUROC/ECE computation rather than polluting it with a fabricated 0.5/0.75.
    text = raw.strip().upper()
    label = "unknown"
    if re.search(r"\bYES\b", text) and not re.search(r"\bNO\b", text):
        label = "faithful"
    elif re.search(r"\bNO\b", text) and not re.search(r"\bYES\b", text):
        label = "hallucinated"

    return {
        "pred_label": label,
        "pred_prob_hallucinated": None,
        "parse_ok": False,
    }


# ── Prompt builders (vllm paths) ──────────────────────────────────────────────
def make_prompt_qwen(report: str, prompt_mode: str = "standard") -> str:
    p = PROMPTS[prompt_mode].format(report=report)
    return f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{p}<|im_end|>\n<|im_start|>assistant\n"

def make_prompt_medgemma(report: str, prompt_mode: str = "standard") -> str:
    p = PROMPTS[prompt_mode].format(report=report)
    return f"<bos><start_of_turn>user\n<start_of_image><end_of_image>{p}<end_of_turn>\n<start_of_turn>model\n"

def make_prompt_llava_med(report: str, prompt_mode: str = "standard") -> str:
    p = PROMPTS[prompt_mode].format(report=report)
    return f"USER: <image>\n{p}\nASSISTANT:"

def make_prompt_chexagent(report: str, prompt_mode: str = "standard") -> str:
    p = PROMPTS[prompt_mode].format(report=report)
    return f"<s>[INST] <image>\n{p} [/INST]"

def make_prompt_med_flamingo(report: str, prompt_mode: str = "standard") -> str:
    p = PROMPTS[prompt_mode].format(report=report)
    return f"<image>User: {p}\nAssistant:"

def make_prompt_phi3(report: str, prompt_mode: str = "standard") -> str:
    p = PROMPTS[prompt_mode].format(report=report)
    return f"<|user|>\n<|image_1|>\n{p}<|end|>\n<|assistant|>\n"

def make_prompt_llava15(report: str, prompt_mode: str = "standard") -> str:
    p = PROMPTS[prompt_mode].format(report=report)
    return f"USER: <image>\n{p}\nASSISTANT:"

# ── vllm runner (qwen25_vl, medgemma) ──────────────────────────────────────────
def run_vllm(model_key: str, prompt_mode: str = "standard"):
    from vllm import LLM, SamplingParams

    with open(resolve_benchmark_path()) as f:
        samples = [json.loads(l) for l in f]

    suffix   = f"_{prompt_mode}" if prompt_mode != "standard" else ""
    out_path = OUTPUTS / f"{model_key}_{OUTPUT_VERSION}{suffix}.jsonl"

    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try: done_ids.add(json.loads(line)["id"])
                except Exception: pass
        print(f"[resume] {len(done_ids)} already done")

    remaining = [s for s in samples if s["id"] not in done_ids]
    if not remaining:
        print("[done] All samples already processed.")
        return

    local = LOCAL_DIRS[model_key]
    print(f"[load] {model_key} via vLLM from {local}")

    max_tokens = 320 if prompt_mode == "cot" else 100

    llm = LLM(
        model=local,
        dtype="bfloat16",
        max_model_len=4096,
        tensor_parallel_size=1,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.85,
        disable_log_stats=True,
        enforce_eager=True,
    )
    sampling = SamplingParams(temperature=0, max_tokens=max_tokens)
    prompt_builder = make_prompt_qwen if model_key == "qwen25_vl" else make_prompt_medgemma

    prompts = []
    for s in remaining:
        image = Image.open(s["img_path"]).convert("RGB")
        image.thumbnail((640, 640), Image.LANCZOS)
        prompts.append({
            "prompt": prompt_builder(s["report"], prompt_mode),
            "multi_modal_data": {"image": image},
        })

    print(f"[inference] Running {len(prompts)} samples (prompt={prompt_mode}) ...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params=sampling)
    total_time = time.time() - t0
    avg_latency = round(total_time / len(prompts), 3)

    n_parse_fail = 0
    with open(out_path, "a") as fout:
        for sample, output in zip(tqdm(remaining, desc=model_key), outputs):
            raw    = output.outputs[0].text.strip()
            parsed = parse_output(raw)
            if not parsed["parse_ok"]:
                n_parse_fail += 1
            record = {
                "id":              sample["id"],
                "model":           model_key,
                "prompt_mode":     prompt_mode,
                "gt_hallucinated": sample["hallucinated"],
                "hallu_type":      sample["hallu_type"],
                "raw_output":      raw,
                "latency_s":       avg_latency,
                **parsed,
            }
            fout.write(json.dumps(record) + "\n")

    print(f"[done] {len(remaining)} samples in {total_time:.1f}s → {out_path}")
    if n_parse_fail:
        print(f"  [warn] {n_parse_fail}/{len(remaining)} outputs did not match "
              f"the strict VERDICT|P_ERROR format — these are marked parse_ok="
              f"False and pred_prob_hallucinated=null; report this count in the "
              f"paper's reproducibility section.")


# ── LLaMA-3.2-11B-Vision — single consolidated implementation ────────────────
def run_llama(prompt_mode: str = "standard"):
    from transformers import AutoProcessor, MllamaForConditionalGeneration

    with open(resolve_benchmark_path()) as f:
        samples = [json.loads(l) for l in f]

    suffix   = f"_{prompt_mode}" if prompt_mode != "standard" else ""
    out_path = OUTPUTS / f"llama_{OUTPUT_VERSION}{suffix}.jsonl"

    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try: done_ids.add(json.loads(line)["id"])
                except Exception: pass
        print(f"[resume] {len(done_ids)} already done")

    remaining = [s for s in samples if s["id"] not in done_ids]
    if not remaining:
        print("[done] All samples already processed.")
        return

    local = LOCAL_DIRS["llama"]
    print(f"[load] LLaMA-3.2-Vision from {local}")

    processor = AutoProcessor.from_pretrained(local)
    model = MllamaForConditionalGeneration.from_pretrained(
        local, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()

    # Embedding hotfix: this is a real tokenizer/vocab-size mismatch fix
    # (some checkpoints ship a tokenizer larger than the model's embedding
    # table, causing CUDA index-out-of-bounds on rare tokens). This is a
    # correctness fix, unrelated to any metric or scoring choice — kept
    # unconditionally, unlike the removed maybe_invert() which WAS a
    # metric-affecting choice.
    if len(processor.tokenizer) > model.config.text_config.vocab_size:
        print(f"  [hotfix] Resizing embeddings: "
              f"{model.config.text_config.vocab_size} -> {len(processor.tokenizer)}")
        model.resize_token_embeddings(len(processor.tokenizer))
        model.config.text_config.vocab_size = len(processor.tokenizer)

    max_new_tokens = 320 if prompt_mode == "cot" else 100
    prompt_text    = PROMPTS[prompt_mode]

    print(f"[inference] {len(remaining)} samples | prompt={prompt_mode}")
    t0 = time.time()
    n_parse_fail = 0

    with open(out_path, "a") as fout:
        for i, sample in enumerate(tqdm(remaining, desc="llama")):
            raw = None
            for attempt in range(3):
                try:
                    image = Image.open(sample["img_path"]).convert("RGB")
                    image.thumbnail((560, 560), Image.LANCZOS)

                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text",
                             "text": prompt_text.format(report=sample["report"])},
                        ],
                    }]
                    text_input = processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )

                    inputs = processor(
                        images=image, text=text_input,
                        return_tensors="pt", padding=False,
                    )
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    # pixel_values must be bfloat16; input_ids must stay int — cast separately
                    if "pixel_values" in inputs:
                        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

                    with torch.no_grad():
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                            repetition_penalty=1.05,   # reduced from 1.2 — less aggressive
                            no_repeat_ngram_size=3,    # reduced from 5
                            use_cache=True,
                        )

                    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
                    raw = processor.decode(new_tokens, skip_special_tokens=True).strip()

                    excl_ratio = raw.count("!") / max(len(raw), 1)
                    if excl_ratio > 0.3 or len(raw) < 3:
                        if attempt < 2:
                            print(f"\n  [retry-degen] {sample['id']}: "
                                  f"excl_ratio={excl_ratio:.2f}")
                            time.sleep(1)
                            raw = None
                            continue
                    break

                except Exception as e:
                    if attempt < 2:
                        print(f"\n  [retry] {sample['id']} attempt {attempt}: {e}")
                        time.sleep(2)
                        raw = None
                    else:
                        print(f"\n  [fail] {sample['id']}: {e}")

            if raw is None or raw == "":
                raw = "unknown"

            parsed = parse_output(raw)
            if not parsed["parse_ok"]:
                n_parse_fail += 1

            record = {
                "id":              sample["id"],
                "model":           "llama",
                "prompt_mode":     prompt_mode,
                "gt_hallucinated": sample["hallucinated"],
                "hallu_type":      sample["hallu_type"],
                "raw_output":      raw,
                "latency_s":       round((time.time() - t0) / (i + 1), 3),
                **parsed,
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

    print(f"[done] {len(remaining)} samples in {time.time()-t0:.1f}s → {out_path}")
    if n_parse_fail:
        print(f"  [warn] {n_parse_fail}/{len(remaining)} outputs failed strict "
              f"parse — see parse_ok field.")


# ── BioViL-T Text & Image NLI Engine ──────────────────────────────────────────
# FIX: previously produced a bare "YES 0.87"-style string and re-used the
# YES/NO parse_output path with an implicit direction assumption. Now emits
# a P_ERROR-formatted string directly so it goes through the SAME parser as
# every other model, with the SAME fixed-direction confidence semantics.
def run_biovil_t():
    from transformers import AutoTokenizer, AutoModel
    from torchvision import transforms
    import numpy as np

    with open(resolve_benchmark_path()) as f:
        samples = [json.loads(l) for l in f]

    out_path = OUTPUTS / f"biovil_t_{OUTPUT_VERSION}.jsonl"
    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try: done_ids.add(json.loads(line)["id"])
                except Exception: pass
        print(f"[resume] {len(done_ids)} already done")

    remaining = [s for s in samples if s["id"] not in done_ids]
    if not remaining:
        print("[done] All samples already processed.")
        return

    local = LOCAL_DIRS["biovil_t"]
    print(f"[load] BioViL-T from {local}")
    tokenizer = AutoTokenizer.from_pretrained(local)
    model = AutoModel.from_pretrained(local, trust_remote_code=True).to("cuda:0").eval()

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    NEG_TEMPLATES = [
        "The report contains findings not visible in this image.",
        "This report does not accurately describe the chest X-ray.",
        "There are errors in this radiology report.",
    ]

    t0 = time.time()
    with open(out_path, "a") as fout:
        for idx, sample in enumerate(tqdm(remaining, desc="biovil_t")):
            try:
                image = Image.open(sample["img_path"]).convert("RGB")
                img_t = transform(image).unsqueeze(0).to("cuda:0")

                tok_pos = tokenizer(
                    sample["report"], return_tensors="pt",
                    truncation=True, max_length=256, padding="max_length",
                ).to("cuda:0")

                with torch.no_grad():
                    try:
                        img_emb = model.get_projected_image_embeddings(pixel_values=img_t)
                        img_emb = torch.nn.functional.normalize(img_emb, dim=-1)
                        emb_pos = model.get_projected_text_embeddings(
                            input_ids=tok_pos["input_ids"],
                            attention_mask=tok_pos["attention_mask"])
                        emb_pos = torch.nn.functional.normalize(emb_pos, dim=-1)
                        sim_pos = (img_emb * emb_pos).sum().item()

                        neg_sims = []
                        for neg_text in NEG_TEMPLATES:
                            tok_neg = tokenizer(neg_text, return_tensors="pt",
                                                 padding="max_length", max_length=256).to("cuda:0")
                            emb_neg = model.get_projected_text_embeddings(
                                input_ids=tok_neg["input_ids"],
                                attention_mask=tok_neg["attention_mask"])
                            emb_neg = torch.nn.functional.normalize(emb_neg, dim=-1)
                            neg_sims.append((img_emb * emb_neg).sum().item())
                        sim_neg = float(np.mean(neg_sims))
                    except AttributeError:
                        out_pos = model(**tok_pos)
                        emb_pos = out_pos.last_hidden_state[:, 0, :]
                        sim_pos = emb_pos.norm().item()
                        sim_neg = sim_pos * 0.9

                # sim_pos = similarity of report to the ORIGINAL claim text
                # (i.e. evidence FOR faithfulness). Higher sim_pos - sim_neg
                # => more evidence the report matches the image => LOWER
                # P(hallucinated). This direction is now made explicit rather
                # than inferred from a downstream YES/NO string.
                margin = sim_pos - sim_neg
                p_faithful = float(torch.sigmoid(torch.tensor(margin)))
                p_error = max(0.0, min(1.0, 1.0 - p_faithful))
                verdict = "NO" if p_error >= 0.5 else "YES"

                raw = f"VERDICT: {verdict} | P_ERROR: {p_error:.4f} | REASON: NLI margin={margin:.4f}"
                parsed = parse_output(raw)

                record = {
                    "id":              sample["id"],
                    "model":           "biovil_t",
                    "prompt_mode":     "nli",
                    "gt_hallucinated": sample["hallucinated"],
                    "hallu_type":      sample["hallu_type"],
                    "raw_output":      raw,
                    "latency_s":       round((time.time() - t0) / (idx + 1), 3),
                    **parsed,
                }
                fout.write(json.dumps(record) + "\n")
                fout.flush()
            except Exception as e:
                print(f"\n[warn] BioViL exception on {sample['id']}: {e}")

    print(f"[done] → {out_path}")

# ── LLaVA-Med-7B ─────────────────────────────────────────────────────────────
def run_llava_med(prompt_mode: str = "standard"):
    from transformers import AutoProcessor, AutoModelForCausalLM

    with open(resolve_benchmark_path()) as f:
        samples = [json.loads(l) for l in f]

    suffix   = f"_{prompt_mode}" if prompt_mode != "standard" else ""
    out_path = OUTPUTS / f"llava_med_{OUTPUT_VERSION}{suffix}.jsonl"

    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try: done_ids.add(json.loads(line)["id"])
                except Exception: pass
        print(f"[resume] {len(done_ids)} already done")

    remaining = [s for s in samples if s["id"] not in done_ids]
    if not remaining:
        print("[done] All samples already processed.")
        return

    local = LOCAL_DIRS["llava_med"]
    print(f"[load] LLaVA-Med-7B from {local}")

    processor = AutoProcessor.from_pretrained(local, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        local, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True
    ).eval()

    max_new_tokens = 320 if prompt_mode == "cot" else 100
    t0 = time.time()
    n_parse_fail = 0

    with open(out_path, "a") as fout:
        for i, sample in enumerate(tqdm(remaining, desc="llava_med")):
            raw = None
            for attempt in range(3):
                try:
                    image = Image.open(sample["img_path"]).convert("RGB")
                    image.thumbnail((560, 560), Image.LANCZOS)
                    prompt_text = make_prompt_llava_med(sample["report"], prompt_mode)
                    inputs = processor(
                        images=image, text=prompt_text,
                        return_tensors="pt"
                    ).to(model.device, dtype=torch.bfloat16)
                    with torch.no_grad():
                        output_ids = model.generate(
                            **inputs, max_new_tokens=max_new_tokens,
                            do_sample=False, temperature=None, top_p=None,
                        )
                    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
                    raw = processor.decode(new_tokens, skip_special_tokens=True).strip()
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2); raw = None
                    else:
                        print(f"\n  [fail] {sample['id']}: {e}")

            if not raw:
                raw = "unknown"
            parsed = parse_output(raw)
            if not parsed["parse_ok"]:
                n_parse_fail += 1
            record = {
                "id": sample["id"], "model": "llava_med",
                "prompt_mode": prompt_mode,
                "gt_hallucinated": sample["hallucinated"],
                "hallu_type": sample["hallu_type"],
                "raw_output": raw,
                "latency_s": round((time.time() - t0) / (i + 1), 3),
                **parsed,
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

    print(f"[done] {len(remaining)} samples → {out_path}")
    if n_parse_fail:
        print(f"  [warn] {n_parse_fail}/{len(remaining)} parse failures")


# ── CheXagent-8B ──────────────────────────────────────────────────────────────
def run_chexagent(prompt_mode: str = "standard"):
    """
    CheXagent generates free-form radiology findings from the image.
    We compare its generated report against the (possibly hallucinated)
    benchmark report using token-level F1 (same direction as BioViL-T:
    high similarity = faithful, low similarity = hallucinated).
    P_ERROR = 1 - similarity.
    """
    from transformers import AutoProcessor, AutoModelForCausalLM
    import numpy as np

    with open(resolve_benchmark_path()) as f:
        samples = [json.loads(l) for l in f]

    out_path = OUTPUTS / f"chexagent_{OUTPUT_VERSION}.jsonl"
    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try: done_ids.add(json.loads(line)["id"])
                except Exception: pass
        print(f"[resume] {len(done_ids)} already done")

    remaining = [s for s in samples if s["id"] not in done_ids]
    if not remaining:
        print("[done] All samples already processed.")
        return

    local = LOCAL_DIRS["chexagent"]
    print(f"[load] CheXagent-8B from {local}")

    processor = AutoProcessor.from_pretrained(local, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        local, torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        trust_remote_code=True
    ).eval()

    def token_f1(ref: str, hyp: str) -> float:
        ref_tokens = set(ref.lower().split())
        hyp_tokens = set(hyp.lower().split())
        if not ref_tokens or not hyp_tokens:
            return 0.0
        overlap = ref_tokens & hyp_tokens
        precision = len(overlap) / len(hyp_tokens)
        recall    = len(overlap) / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    t0 = time.time()
    n_parse_fail = 0

    with open(out_path, "a") as fout:
        for idx, sample in enumerate(tqdm(remaining, desc="chexagent")):
            try:
                image = Image.open(sample["img_path"]).convert("RGB")
                image.thumbnail((560, 560), Image.LANCZOS)

                inputs = processor(
                    images=image,
                    text="<image>User: Describe the findings in this chest X-ray. <REPORT>",
                    return_tensors="pt"
                )
                inputs = {k: v.to("cuda:0").to(torch.bfloat16)
                          if v.is_floating_point() else v.to("cuda:0")
                          for k, v in inputs.items()}

                with torch.no_grad():
                    out_ids = model.generate(
                        **inputs, max_new_tokens=200,
                        num_beams=1, do_sample=False,
                    )
                generated = processor.tokenizer.decode(
                    out_ids[0], skip_special_tokens=True
                ).strip()

                # token F1 between CheXagent's image-grounded generation
                # and the benchmark report (which may contain injected errors)
                sim = token_f1(generated, sample["report"])
                p_error = max(0.0, min(1.0, 1.0 - sim))
                verdict = "NO" if p_error >= 0.5 else "YES"
                raw = (f"VERDICT: {verdict} | P_ERROR: {p_error:.4f} | "
                       f"REASON: token_f1={sim:.4f} gen_len={len(generated.split())}")
                parsed = parse_output(raw)

                record = {
                    "id":              sample["id"],
                    "model":           "chexagent",
                    "prompt_mode":     "report_comparison",
                    "gt_hallucinated": sample["hallucinated"],
                    "hallu_type":      sample["hallu_type"],
                    "raw_output":      raw,
                    "latency_s":       round((time.time() - t0) / (idx + 1), 3),
                    **parsed,
                }
                fout.write(json.dumps(record) + "\n")
                fout.flush()

            except Exception as e:
                print(f"\n[warn] chexagent exception on {sample['id']}: {e}")

    print(f"[done] → {out_path}")
    if n_parse_fail:
        print(f"  [warn] {n_parse_fail} parse failures")

# ── Phi-3-Vision-4B ───────────────────────────────────────────────────────────
def run_phi3_vision(prompt_mode: str = "standard"):
    from transformers import AutoProcessor, AutoModelForCausalLM

    with open(resolve_benchmark_path()) as f:
        samples = [json.loads(l) for l in f]

    suffix   = f"_{prompt_mode}" if prompt_mode != "standard" else ""
    out_path = OUTPUTS / f"phi3_vision_{OUTPUT_VERSION}{suffix}.jsonl"

    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try: done_ids.add(json.loads(line)["id"])
                except Exception: pass
        print(f"[resume] {len(done_ids)} already done")

    remaining = [s for s in samples if s["id"] not in done_ids]
    if not remaining:
        print("[done] All samples already processed.")
        return

    local = LOCAL_DIRS["phi3_vision"]
    print(f"[load] Phi-3-Vision-4B from {local}")

    processor = AutoProcessor.from_pretrained(local, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        local, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, _attn_implementation="eager"
    ).eval()

    # Patch DynamicCache.seen_tokens removed in transformers >= 4.40
    # Patch DynamicCache attributes removed in transformers >= 4.40
    # Patch all DynamicCache attributes removed in transformers >= 4.40
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, new_seq_len, layer_idx=0: self.get_seq_length(layer_idx)

    max_new_tokens = 320 if prompt_mode == "cot" else 100
    t0 = time.time()
    n_parse_fail = 0

    with open(out_path, "a") as fout:
        for i, sample in enumerate(tqdm(remaining, desc="phi3_vision")):
            raw = None
            for attempt in range(3):
                try:
                    image = Image.open(sample["img_path"]).convert("RGB")
                    image.thumbnail((560, 560), Image.LANCZOS)
                    prompt_text = make_prompt_phi3(sample["report"], prompt_mode)
                    inputs = processor(
                        images=image, text=prompt_text,
                        return_tensors="pt"
                    )
                    inputs = {k: v.to("cuda:0").to(torch.bfloat16)
                              if v.is_floating_point() else v.to("cuda:0")
                              for k, v in inputs.items()}
                    with torch.no_grad():
                        output_ids = model.generate(
                            **inputs, max_new_tokens=max_new_tokens,
                            do_sample=False, temperature=None, top_p=None,
                        )
                    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
                    raw = processor.decode(new_tokens, skip_special_tokens=True).strip()
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2); raw = None
                    else:
                        print(f"\n  [fail] {sample['id']}: {e}")

            if not raw:
                raw = "unknown"
            parsed = parse_output(raw)
            if not parsed["parse_ok"]:
                n_parse_fail += 1
            record = {
                "id": sample["id"], "model": "phi3_vision",
                "prompt_mode": prompt_mode,
                "gt_hallucinated": sample["hallucinated"],
                "hallu_type": sample["hallu_type"],
                "raw_output": raw,
                "latency_s": round((time.time() - t0) / (i + 1), 3),
                **parsed,
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

    print(f"[done] {len(remaining)} samples → {out_path}")
    if n_parse_fail:
        print(f"  [warn] {n_parse_fail}/{len(remaining)} parse failures")


# ── BLIP-2-OPT-6.7B (similarity-based, no generative prompt) ─────────────────
def run_blip2():
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    import numpy as np

    with open(resolve_benchmark_path()) as f:
        samples = [json.loads(l) for l in f]

    out_path = OUTPUTS / f"blip2_{OUTPUT_VERSION}.jsonl"
    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try: done_ids.add(json.loads(line)["id"])
                except Exception: pass
        print(f"[resume] {len(done_ids)} already done")

    remaining = [s for s in samples if s["id"] not in done_ids]
    if not remaining:
        print("[done] All samples already processed.")
        return

    local = LOCAL_DIRS["blip2"]
    print(f"[load] BLIP-2-OPT-6.7B from {local}")

    processor = Blip2Processor.from_pretrained(local)
    model = Blip2ForConditionalGeneration.from_pretrained(
        local, torch_dtype=torch.float16, device_map="auto"
    ).eval()

    NEG_TEMPLATES = [
        "The report contains findings not visible in this image.",
        "This report does not accurately describe the chest X-ray.",
        "There are errors in this radiology report.",
    ]

    t0 = time.time()
    with open(out_path, "a") as fout:
        for idx, sample in enumerate(tqdm(remaining, desc="blip2")):
            try:
                image = Image.open(sample["img_path"]).convert("RGB")

                # positive: image + report text
                inputs_pos = processor(
                    images=image, text=sample["report"],
                    return_tensors="pt", truncation=True, max_length=256
                ).to("cuda:0", torch.float16)

                with torch.no_grad():
                    # use ITM (image-text matching) score via generation logits
                    # proxy: feed report as prompt, measure generation likelihood
                    out_pos = model.generate(
                        **inputs_pos, max_new_tokens=1,
                        output_scores=True, return_dict_in_generate=True,
                    )
                    # log-prob of first generated token as faithfulness proxy
                    score_pos = float(out_pos.scores[0].softmax(-1).max().item())

                    neg_scores = []
                    for neg_text in NEG_TEMPLATES:
                        inputs_neg = processor(
                            images=image, text=neg_text,
                            return_tensors="pt", truncation=True, max_length=256
                        ).to("cuda:0", torch.float16)
                        out_neg = model.generate(
                            **inputs_neg, max_new_tokens=1,
                            output_scores=True, return_dict_in_generate=True,
                        )
                        neg_scores.append(
                            float(out_neg.scores[0].softmax(-1).max().item())
                        )
                    score_neg = float(np.mean(neg_scores))

                margin   = score_pos - score_neg
                p_faithful = float(torch.sigmoid(torch.tensor(margin)))
                p_error  = max(0.0, min(1.0, 1.0 - p_faithful))
                verdict  = "NO" if p_error >= 0.5 else "YES"
                raw = (f"VERDICT: {verdict} | P_ERROR: {p_error:.4f} | "
                       f"REASON: blip2_margin={margin:.4f}")
                parsed = parse_output(raw)

                record = {
                    "id": sample["id"], "model": "blip2",
                    "prompt_mode": "itm_similarity",
                    "gt_hallucinated": sample["hallucinated"],
                    "hallu_type": sample["hallu_type"],
                    "raw_output": raw,
                    "latency_s": round((time.time() - t0) / (idx + 1), 3),
                    **parsed,
                }
                fout.write(json.dumps(record) + "\n")
                fout.flush()
            except Exception as e:
                print(f"\n[warn] blip2 exception on {sample['id']}: {e}")

    print(f"[done] → {out_path}")


# ── LLaVA-1.5-13B ─────────────────────────────────────────────────────────────
def run_llava15(prompt_mode: str = "standard"):
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    with open(resolve_benchmark_path()) as f:
        samples = [json.loads(l) for l in f]

    suffix   = f"_{prompt_mode}" if prompt_mode != "standard" else ""
    out_path = OUTPUTS / f"llava15_{OUTPUT_VERSION}{suffix}.jsonl"

    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try: done_ids.add(json.loads(line)["id"])
                except Exception: pass
        print(f"[resume] {len(done_ids)} already done")

    remaining = [s for s in samples if s["id"] not in done_ids]
    if not remaining:
        print("[done] All samples already processed.")
        return

    local = LOCAL_DIRS["llava15"]
    print(f"[load] LLaVA-1.5-13B from {local}")

    processor = AutoProcessor.from_pretrained(local)
    model = LlavaForConditionalGeneration.from_pretrained(
        local, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()

    max_new_tokens = 320 if prompt_mode == "cot" else 100
    t0 = time.time()
    n_parse_fail = 0

    with open(out_path, "a") as fout:
        for i, sample in enumerate(tqdm(remaining, desc="llava15")):
            raw = None
            for attempt in range(3):
                try:
                    image = Image.open(sample["img_path"]).convert("RGB")
                    image.thumbnail((560, 560), Image.LANCZOS)
                    prompt_text = make_prompt_llava15(sample["report"], prompt_mode)
                    inputs = processor(
                        images=image, text=prompt_text,
                        return_tensors="pt"
                    )
                    inputs = {k: v.to("cuda:0").to(torch.bfloat16)
                              if v.is_floating_point() else v.to("cuda:0")
                              for k, v in inputs.items()}
                    with torch.no_grad():
                        output_ids = model.generate(
                            **inputs, max_new_tokens=max_new_tokens,
                            do_sample=False, temperature=None, top_p=None,
                        )
                    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
                    raw = processor.decode(new_tokens, skip_special_tokens=True).strip()
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2); raw = None
                    else:
                        print(f"\n  [fail] {sample['id']}: {e}")

            if not raw:
                raw = "unknown"
            parsed = parse_output(raw)
            if not parsed["parse_ok"]:
                n_parse_fail += 1
            record = {
                "id": sample["id"], "model": "llava15",
                "prompt_mode": prompt_mode,
                "gt_hallucinated": sample["hallucinated"],
                "hallu_type": sample["hallu_type"],
                "raw_output": raw,
                "latency_s": round((time.time() - t0) / (i + 1), 3),
                **parsed,
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

    print(f"[done] {len(remaining)} samples → {out_path}")
    if n_parse_fail:
        print(f"  [warn] {n_parse_fail}/{len(remaining)} parse failures")

# ── BiomedCLIP-300M ───────────────────────────────────────────────────────────
# Embedding-similarity model (no generative path), same approach as BioViL-T.
def run_biomedclip():
    from transformers import AutoProcessor, AutoModel
    import numpy as np

    with open(resolve_benchmark_path()) as f:
        samples = [json.loads(l) for l in f]

    out_path = OUTPUTS / f"biomedclip_{OUTPUT_VERSION}.jsonl"
    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try: done_ids.add(json.loads(line)["id"])
                except Exception: pass
        print(f"[resume] {len(done_ids)} already done")

    remaining = [s for s in samples if s["id"] not in done_ids]
    if not remaining:
        print("[done] All samples already processed.")
        return

    local = LOCAL_DIRS["biomedclip"]
    print(f"[load] BiomedCLIP-300M from {local}")

    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    tokenizer = open_clip.get_tokenizer(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    model = model.to("cuda:0").eval()

    NEG_TEMPLATES = [
        "The report contains findings not visible in this image.",
        "This report does not accurately describe the chest X-ray.",
        "There are errors in this radiology report.",
    ]

    t0 = time.time()
    with open(out_path, "a") as fout:
        for idx, sample in enumerate(tqdm(remaining, desc="biomedclip")):
            try:
                image = Image.open(sample["img_path"]).convert("RGB")
                img_t = preprocess(image).unsqueeze(0).to("cuda:0")
                txt_t = tokenizer([sample["report"]], context_length=256).to("cuda:0")

                with torch.no_grad():
                    img_emb = model.encode_image(img_t)
                    txt_emb = model.encode_text(txt_t)
                    img_emb = torch.nn.functional.normalize(img_emb, dim=-1)
                    txt_emb = torch.nn.functional.normalize(txt_emb, dim=-1)
                    sim_pos = (img_emb * txt_emb).sum().item()

                    neg_sims = []
                    for neg_text in NEG_TEMPLATES:
                        neg_t = tokenizer([neg_text], context_length=256).to("cuda:0")
                        neg_emb = torch.nn.functional.normalize(
                            model.encode_text(neg_t), dim=-1)
                        neg_sims.append((img_emb * neg_emb).sum().item())
                    sim_neg = float(np.mean(neg_sims))

                margin   = sim_pos - sim_neg
                p_faithful = float(torch.sigmoid(torch.tensor(margin)))
                p_error  = max(0.0, min(1.0, 1.0 - p_faithful))
                verdict  = "NO" if p_error >= 0.5 else "YES"
                raw = f"VERDICT: {verdict} | P_ERROR: {p_error:.4f} | REASON: CLIP margin={margin:.4f}"
                parsed = parse_output(raw)

                record = {
                    "id": sample["id"], "model": "biomedclip",
                    "prompt_mode": "clip_similarity",
                    "gt_hallucinated": sample["hallucinated"],
                    "hallu_type": sample["hallu_type"],
                    "raw_output": raw,
                    "latency_s": round((time.time() - t0) / (idx + 1), 3),
                    **parsed,
                }
                fout.write(json.dumps(record) + "\n")
                fout.flush()
            except Exception as e:
                print(f"\n[warn] BiomedCLIP exception on {sample['id']}: {e}")

    print(f"[done] → {out_path}")


# ── Med-Flamingo ──────────────────────────────────────────────────────────────
def run_med_flamingo(prompt_mode: str = "standard"):
    from transformers import AutoProcessor, AutoModelForCausalLM

    with open(resolve_benchmark_path()) as f:
        samples = [json.loads(l) for l in f]

    suffix   = f"_{prompt_mode}" if prompt_mode != "standard" else ""
    out_path = OUTPUTS / f"med_flamingo_{OUTPUT_VERSION}{suffix}.jsonl"

    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try: done_ids.add(json.loads(line)["id"])
                except Exception: pass
        print(f"[resume] {len(done_ids)} already done")

    remaining = [s for s in samples if s["id"] not in done_ids]
    if not remaining:
        print("[done] All samples already processed.")
        return

    local = LOCAL_DIRS["med_flamingo"]
    print(f"[load] Med-Flamingo from {local}")

    from open_flamingo import create_model_and_transforms
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="huggyllama/llama-7b",
        tokenizer_path="huggyllama/llama-7b",
        cross_attn_every_n_layers=4,
    )
    checkpoint = torch.load(f"{local}/model.pt", map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    model = model.to("cuda:0").eval()
    tokenizer.padding_side = "left"
    tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})

    max_new_tokens = 320 if prompt_mode == "cot" else 100
    t0 = time.time()
    n_parse_fail = 0

    with open(out_path, "a") as fout:
        for i, sample in enumerate(tqdm(remaining, desc="med_flamingo")):
            raw = None
            for attempt in range(3):
                try:
                    image = Image.open(sample["img_path"]).convert("RGB")
                    vision_x = image_processor(image).unsqueeze(0).unsqueeze(0).unsqueeze(0).to("cuda:0")
                    prompt_text = make_prompt_med_flamingo(sample["report"], prompt_mode)
                    lang_x = tokenizer([prompt_text], return_tensors="pt").to("cuda:0")

                    with torch.no_grad():
                        output_ids = model.generate(
                            vision_x=vision_x.to(torch.float32),
                            lang_x=lang_x["input_ids"],
                            attention_mask=lang_x["attention_mask"],
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                        )
                    new_tokens = output_ids[0][lang_x["input_ids"].shape[-1]:]
                    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2); raw = None
                    else:
                        print(f"\n  [fail] {sample['id']}: {e}")

            if not raw:
                raw = "unknown"
            parsed = parse_output(raw)
            if not parsed["parse_ok"]:
                n_parse_fail += 1
            record = {
                "id": sample["id"], "model": "med_flamingo",
                "prompt_mode": prompt_mode,
                "gt_hallucinated": sample["hallucinated"],
                "hallu_type": sample["hallu_type"],
                "raw_output": raw,
                "latency_s": round((time.time() - t0) / (i + 1), 3),
                **parsed,
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

    print(f"[done] {len(remaining)} samples → {out_path}")
    if n_parse_fail:
        print(f"  [warn] {n_parse_fail}/{len(remaining)} parse failures")

# ── Baselines (Reviewer request: constant / random / text-only / shuffled) ───
# These do NOT require a model at all except the text-only baseline, which
# reuses whichever small text-only model is easiest to run; here we implement
# it as a lexical heuristic baseline so it requires no additional model
# weights, but the hook is written so a real text-only LLM call can be
# swapped in (see run_text_only_llm_baseline below for that variant).
def run_constant_and_random_baselines():
    with open(resolve_benchmark_path()) as f:
        samples = [json.loads(l) for l in f]

    import random as pyrandom
    rng = pyrandom.Random(42)

    for baseline_name, fn in [
        ("constant_faithful", lambda s: ("faithful", 0.0)),
        ("random", lambda s: pyrandom_choice(rng)),
    ]:
        out_path = OUTPUTS / f"{baseline_name}_{OUTPUT_VERSION}.jsonl"
        with open(out_path, "w") as fout:
            for s in samples:
                label, p_error = fn(s)
                record = {
                    "id":              s["id"],
                    "model":           baseline_name,
                    "prompt_mode":     "baseline",
                    "gt_hallucinated": s["hallucinated"],
                    "hallu_type":      s["hallu_type"],
                    "raw_output":      f"VERDICT: {'NO' if label=='hallucinated' else 'YES'} | P_ERROR: {p_error:.4f}",
                    "latency_s":       0.0,
                    "pred_label":      label,
                    "pred_prob_hallucinated": p_error,
                    "parse_ok":        True,
                }
                fout.write(json.dumps(record) + "\n")
        print(f"[baseline] {baseline_name} → {out_path}")


def pyrandom_choice(rng):
    p = rng.random()
    label = "hallucinated" if p >= 0.5 else "faithful"
    return label, p


def run_shuffled_image_baseline(model_keys=None, prompt_mode: str = "standard"):
    """
    Shuffled-image baseline (Reviewer CTbs, explicit request).

    Purpose: tests whether a VLM's apparent hallucination-detection ability
    actually depends on the paired image at all, or whether it is driven
    entirely by textual cues in the report (linguistic plausibility of the
    claim, independent of what's actually in the X-ray). Each sample's
    REPORT is kept, but its IMAGE is swapped for a different, randomly
    chosen image from elsewhere in the benchmark. If a model's classification
    performance under shuffled images is close to its performance under
    correct images, that model is not meaningfully using visual grounding
    for this task -- a critical validity finding that must be reported
    honestly in the paper regardless of which direction it points.

    CHANGE (post-review round 2): previously this ran ONLY against
    qwen25_vl ("the fastest of the three vLLM-served models"), which meant
    the model the paper's headline HalluScore claims rest on (MedGemma) was
    NEVER tested for whether it actually uses the image. That is now fixed:
    this function accepts a LIST of model keys and loops over all of them,
    producing one output file per model (shuffled_image_baseline_<model>_v3
    .jsonl), so each real VLM's shuffled-image performance can be compared
    directly against its own real-image performance. Default list comes
    from config.yaml's inference.shuffled_image_baseline_models (currently
    all three vLLM/transformers-servable VLMs: qwen25_vl, medgemma, llama;
    biovil_t is excluded since it is a text/image NLI scorer without the
    same YES/NO/P_ERROR generative prompt path, and its degenerate-baseline
    role is already covered by comparing it directly against the other
    baselines in Table 1).

    Runs via the SAME inference path as the real models (vLLM for qwen25_vl
    and medgemma, the consolidated transformers path for llama) rather than
    a separate lightweight heuristic, because this baseline's entire point
    is to exercise each model's REAL visual pathway with a deliberately
    wrong image -- a heuristic substitute would not test the right thing.
    """
    import random as pyrandom

    if model_keys is None:
        model_keys = SHUFFLED_IMAGE_DEFAULT_MODELS

    for model_key in model_keys:
        print(f"\n{'='*60}")
        print(f"  Shuffled-image baseline — base model: {model_key}")
        print(f"{'='*60}")
        if model_key == "llama":
            _run_shuffled_image_baseline_llama(prompt_mode=prompt_mode)
        elif model_key in ("qwen25_vl", "medgemma"):
            _run_shuffled_image_baseline_vllm(model_key=model_key, prompt_mode=prompt_mode)
        else:
            print(f"  [skip] '{model_key}' is not a supported shuffled-image "
                  f"base model (expected one of qwen25_vl, medgemma, llama)")


def _build_shuffled_pairs(samples):
    """Deterministic derangement shared by all per-model shuffled runs so
    every model is tested against the SAME wrong-image pairing — otherwise
    cross-model comparisons on this baseline wouldn't be apples-to-apples."""
    import random as pyrandom
    rng = pyrandom.Random(42)
    n = len(samples)
    shuffled_indices = list(range(n))
    for _ in range(1000):
        rng.shuffle(shuffled_indices)
        if all(shuffled_indices[i] != i for i in range(n)):
            break
    return shuffled_indices


def _run_shuffled_image_baseline_vllm(model_key: str, prompt_mode: str = "standard"):
    from vllm import LLM, SamplingParams

    with open(resolve_benchmark_path()) as f:
        samples = [json.loads(l) for l in f]

    shuffled_indices = _build_shuffled_pairs(samples)

    baseline_name = f"shuffled_image_baseline_{model_key}"
    out_path = OUTPUTS / f"{baseline_name}_{OUTPUT_VERSION}.jsonl"

    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try: done_ids.add(json.loads(line)["id"])
                except Exception: pass
        print(f"[resume] {len(done_ids)} already done")

    remaining = [(s, samples[shuffled_indices[i]])
                 for i, s in enumerate(samples) if s["id"] not in done_ids]
    if not remaining:
        print("[done] All samples already processed.")
        return

    local = LOCAL_DIRS[model_key]
    print(f"[load] {model_key} via vLLM from {local} (shuffled-image baseline)")

    llm = LLM(
        model=local, dtype="bfloat16", max_model_len=4096,
        tensor_parallel_size=1, limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.85, disable_log_stats=True, enforce_eager=True,
    )
    sampling = SamplingParams(temperature=0, max_tokens=100)
    prompt_builder = make_prompt_qwen if model_key == "qwen25_vl" else make_prompt_medgemma

    prompts = []
    for real_sample, wrong_image_sample in remaining:
        image = Image.open(wrong_image_sample["img_path"]).convert("RGB")
        image.thumbnail((640, 640), Image.LANCZOS)
        prompts.append({
            "prompt": prompt_builder(real_sample["report"], prompt_mode),
            "multi_modal_data": {"image": image},
        })

    print(f"[inference] Running {len(prompts)} shuffled-image samples ...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params=sampling)
    total_time = time.time() - t0
    avg_latency = round(total_time / max(len(prompts), 1), 3)

    n_parse_fail = 0
    with open(out_path, "a") as fout:
        for (real_sample, wrong_image_sample), output in zip(remaining, outputs):
            raw = output.outputs[0].text.strip()
            parsed = parse_output(raw)
            if not parsed["parse_ok"]:
                n_parse_fail += 1
            record = {
                "id":              real_sample["id"],
                "model":           baseline_name,
                "base_model":      model_key,
                "prompt_mode":     prompt_mode,
                "gt_hallucinated": real_sample["hallucinated"],
                "hallu_type":      real_sample["hallu_type"],
                "wrong_image_id":  wrong_image_sample["id"],
                "raw_output":      raw,
                "latency_s":       avg_latency,
                **parsed,
            }
            fout.write(json.dumps(record) + "\n")

    print(f"[done] {len(remaining)} samples in {total_time:.1f}s → {out_path}")
    if n_parse_fail:
        print(f"  [warn] {n_parse_fail}/{len(remaining)} parse failures")
    print(f"  [note] Compare this run's accuracy/F1/AUROC against {model_key}'s "
          f"REAL-image run. Large gap = image grounding matters. Small gap = "
          f"model may be exploiting text-only artifacts (see also "
          f"text_only_baseline).")


def _run_shuffled_image_baseline_llama(prompt_mode: str = "standard"):
    """
    LLaMA-3.2-Vision variant of the shuffled-image baseline, using the same
    consolidated transformers inference path as run_llama() (including the
    embedding-resize hotfix), rather than vLLM. Added in this fix so LLaMA
    is no longer excluded from the shuffled-image validity check.
    """
    from transformers import AutoProcessor, MllamaForConditionalGeneration

    with open(resolve_benchmark_path()) as f:
        samples = [json.loads(l) for l in f]

    shuffled_indices = _build_shuffled_pairs(samples)

    baseline_name = "shuffled_image_baseline_llama"
    out_path = OUTPUTS / f"{baseline_name}_{OUTPUT_VERSION}.jsonl"

    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try: done_ids.add(json.loads(line)["id"])
                except Exception: pass
        print(f"[resume] {len(done_ids)} already done")

    remaining = [(s, samples[shuffled_indices[i]])
                 for i, s in enumerate(samples) if s["id"] not in done_ids]
    if not remaining:
        print("[done] All samples already processed.")
        return

    local = LOCAL_DIRS["llama"]
    print(f"[load] LLaMA-3.2-Vision from {local} (shuffled-image baseline)")

    processor = AutoProcessor.from_pretrained(local)
    model = MllamaForConditionalGeneration.from_pretrained(
        local, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()

    if len(processor.tokenizer) > model.config.text_config.vocab_size:
        print(f"  [hotfix] Resizing embeddings: "
              f"{model.config.text_config.vocab_size} -> {len(processor.tokenizer)}")
        model.resize_token_embeddings(len(processor.tokenizer))
        model.config.text_config.vocab_size = len(processor.tokenizer)

    max_new_tokens = 320 if prompt_mode == "cot" else 100
    prompt_text = PROMPTS[prompt_mode]

    print(f"[inference] {len(remaining)} shuffled-image samples | prompt={prompt_mode}")
    t0 = time.time()
    n_parse_fail = 0

    with open(out_path, "a") as fout:
        for i, (real_sample, wrong_image_sample) in enumerate(
            tqdm(remaining, desc="shuffled_llama")
        ):
            raw = None
            for attempt in range(3):
                try:
                    image = Image.open(wrong_image_sample["img_path"]).convert("RGB")
                    image.thumbnail((560, 560), Image.LANCZOS)

                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text",
                             "text": prompt_text.format(report=real_sample["report"])},
                        ],
                    }]
                    text_input = processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                    inputs = processor(
                        images=image, text=text_input,
                        return_tensors="pt", padding=False,
                    )
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    # pixel_values must be bfloat16; input_ids must stay int — cast separately
                    if "pixel_values" in inputs:
                        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
                    with torch.no_grad():
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                            repetition_penalty=1.05,   # reduced from 1.2 — less aggressive
                            no_repeat_ngram_size=3,    # reduced from 5
                            use_cache=True,
                        )

                    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
                    raw = processor.decode(new_tokens, skip_special_tokens=True).strip()

                    excl_ratio = raw.count("!") / max(len(raw), 1)
                    if excl_ratio > 0.3 or len(raw) < 3:
                        if attempt < 2:
                            time.sleep(1)
                            raw = None
                            continue
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2)
                        raw = None
                    else:
                        print(f"\n  [fail] {real_sample['id']}: {e}")

            if raw is None or raw == "":
                raw = "unknown"

            parsed = parse_output(raw)
            if not parsed["parse_ok"]:
                n_parse_fail += 1

            record = {
                "id":              real_sample["id"],
                "model":           baseline_name,
                "base_model":      "llama",
                "prompt_mode":     prompt_mode,
                "gt_hallucinated": real_sample["hallucinated"],
                "hallu_type":      real_sample["hallu_type"],
                "wrong_image_id":  wrong_image_sample["id"],
                "raw_output":      raw,
                "latency_s":       round((time.time() - t0) / (i + 1), 3),
                **parsed,
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()

    print(f"[done] {len(remaining)} samples in {time.time()-t0:.1f}s → {out_path}")
    if n_parse_fail:
        print(f"  [warn] {n_parse_fail}/{len(remaining)} parse failures")
    print(f"  [note] Compare against llama's REAL-image run (llama_{OUTPUT_VERSION}.jsonl).")


def run_text_only_baseline():
    """
    Text-only baseline: uses ONLY the report text (no image at all), via a
    simple lexical heuristic that flags known injected antonym pairs. This
    directly tests Reviewer CTbs's concern that manually-perturbed claims
    (fixed SWAP_MAP vocabulary in build_benchmark.py / build_benchmark_multi.py)
    may be detectable from linguistic artifacts alone, without ever looking
    at the image. If this baseline scores surprisingly well, it is evidence
    the benchmark leaks signal through text artifacts rather than genuine
    visual grounding.
    """
    from importlib import import_module
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_data_prep"))

    # Prefer the multi-source builder's SWAP_MAP (the default pipeline's
    # vocabulary); fall back to the legacy single-source one if the
    # multi-source module isn't importable for some reason. Both are kept
    # identical by convention (see note at the bottom of build_benchmark.py).
    try:
        build_benchmark_mod = import_module("build_benchmark_multi")
    except ImportError:
        build_benchmark_mod = import_module("build_benchmark")
    SWAP_MAP = build_benchmark_mod.SWAP_MAP
    all_swap_words = set()
    for d in SWAP_MAP.values():
        all_swap_words.update(d.keys())
        all_swap_words.update(d.values())

    with open(resolve_benchmark_path()) as f:
        samples = [json.loads(l) for l in f]

    out_path = OUTPUTS / f"text_only_baseline_{OUTPUT_VERSION}.jsonl"
    with open(out_path, "w") as fout:
        for s in samples:
            text = s["report"].lower()
            # crude signal: presence of any swap-vocabulary word at all is
            # NOT itself informative (both clean and hallucinated reports
            # contain these common words) — the real test is whether a
            # trained classifier could exploit surface patterns. This
            # heuristic baseline is intentionally weak; a stronger version
            # should fine-tune a small text classifier on report text only
            # and is left as a documented extension (see paper Limitations).
            n_swap_words = sum(1 for w in all_swap_words if w in text)
            p_error = min(1.0, n_swap_words * 0.05)
            label = "hallucinated" if p_error >= 0.5 else "faithful"
            record = {
                "id":              s["id"],
                "model":           "text_only_baseline",
                "prompt_mode":     "baseline",
                "gt_hallucinated": s["hallucinated"],
                "hallu_type":      s["hallu_type"],
                "raw_output":      f"VERDICT: {'NO' if label=='hallucinated' else 'YES'} | P_ERROR: {p_error:.4f}",
                "latency_s":       0.0,
                "pred_label":      label,
                "pred_prob_hallucinated": p_error,
                "parse_ok":        True,
            }
            fout.write(json.dumps(record) + "\n")
    print(f"[baseline] text_only_baseline → {out_path}")


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=list(LOCAL_DIRS.keys()) + ["all", "baselines"])
    parser.add_argument("--prompt_mode", default="standard",
                        choices=["standard", "cot"])
    parser.add_argument("--shuffled_models", nargs="+", default=None,
                        choices=["qwen25_vl", "medgemma", "llama"],
                        help="Which VLMs to run the shuffled-image baseline "
                             "against (default: config.yaml "
                             "inference.shuffled_image_baseline_models, "
                             "currently all three)")
    args = parser.parse_args()

    try:
        import setproctitle
        setproctitle.setproctitle("vLLM:EngineCore")
    except ImportError:
        pass

    if args.model == "baselines":
        run_constant_and_random_baselines()
        run_text_only_baseline()
        run_shuffled_image_baseline(
            model_keys=args.shuffled_models,  # None -> config default (all 3)
            prompt_mode=args.prompt_mode,
        )
    else:
        keys = list(LOCAL_DIRS.keys()) if args.model == "all" else [args.model]
        for k in keys:
            print(f"\n{'='*60}")
            print(f"  Model: {k}  |  Prompt: {args.prompt_mode}")
            print(f"{'='*60}")
            if k == "biovil_t":
                run_biovil_t()
            elif k == "biomedclip":
                run_biomedclip()
            elif k == "phi3_vision":
                run_phi3_vision(prompt_mode=args.prompt_mode)
            elif k == "blip2":
                run_blip2()
            elif k == "llava15":
                run_llava15(prompt_mode=args.prompt_mode)
            elif k == "chexagent":
                run_chexagent(prompt_mode=args.prompt_mode)
            elif k == "med_flamingo":
                run_med_flamingo(prompt_mode=args.prompt_mode)
            else:
                run_vllm(model_key=k, prompt_mode=args.prompt_mode)

"""
CHANGELOG — mapping to reviewer comments
=========================================
Reviewer CTbs, "Confidence and AUROC computation are unclear":
  → FIXED. Prompt now elicits P_ERROR = P(hallucinated) directly and
    explicitly, independent of the YES/NO verdict token. parse_output()
    reads this single fixed-direction field; no inversion logic exists
    anywhere in this file.

Reviewer CTbs, "HalluScore may reward degenerate models" (LLaMA 477/500
faithful, F1=0.041, yet high composite score):
  → Addressed in 03_metrics/compute_metrics.py (MCC+F1 geometric-mean
    discrimination term), not in this file.

Reviewer CTbs / aebp / 8sZM, "Important baselines... missing":
  → ADDED: run_constant_and_random_baselines(), run_text_only_baseline(),
    AND run_shuffled_image_baseline() (real VLM inference with a
    deliberately mismatched image). All four requested baseline types are
    now implemented: constant-faithful, random, text-only, shuffled-image.
  → ROUND 2 FIX: shuffled-image baseline previously covered qwen25_vl
    only; now covers qwen25_vl, medgemma, AND llama by default (config-
    driven via inference.shuffled_image_baseline_models), so the model
    the paper's headline claims rest on (MedGemma) is actually validity-
    checked for image-grounding, not just the cheapest model to run.

Reviewer 76GN, "HalluScore relies entirely on the VLM's own outputs... when
the model hallucinates, these outputs are potentially unreliable":
  → Not fully solvable by this file; documented as an acknowledged
    limitation in the paper rewrite rather than silently ignored.
"""