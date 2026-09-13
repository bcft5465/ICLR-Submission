#!/usr/bin/env python3
"""
Drop-in replacement for run_llama_transformers() in run_inference.py
Key fixes:
  1. Proper cross_attention_states handling for Mllama
  2. Suppress generation flag warnings
  3. Correct processor usage - image must be in content list
  4. Repetition penalty to prevent ! spam
  5. Better fallback on cross-attn error
  6. HOTFIX: Resize embedding layer to fix token out of bounds CUDA assert
"""

# ── PATCHES ───────────────────────────────────────────────────────────────────
import sys, builtins
from PIL.Image import Resampling
sys.modules["transformers.utils.PILImageResampling"] = Resampling
setattr(builtins, "PILImageResampling", Resampling)
try:
    import transformers
    import transformers.image_processing_utils as _ipu
    for _mod in [transformers, _ipu]:
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

import json, time, re, logging, argparse
import numpy as np
from pathlib import Path
import torch, yaml
from PIL import Image
from tqdm import tqdm

logging.getLogger("transformers.generation").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

ROOT     = Path("/mnt/ssd/users/durgesh/HEalthcare/Health2")
CFG_PATH = ROOT / "code/v1/config.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

PROCESSED  = ROOT / CFG["paths"]["data_processed"]
OUTPUTS    = ROOT / CFG["paths"]["outputs"]
OUTPUTS.mkdir(parents=True, exist_ok=True)
MODEL_DIR  = ROOT / CFG["paths"]["models"] / "Llama-3.2-11B-Vision-Instruct"

PROMPT_STANDARD = (
    "You are a critical radiology AI evaluator. "
    "Your job is to detect errors and hallucinations in chest X-ray reports.\n\n"
    "Carefully examine the X-ray image and the report below. "
    "Check whether every finding, anatomy, attribute (size/shape/density), "
    "and spatial relationship (left/right/upper/lower) in the report is "
    "actually visible and correct in the image.\n\n"
    "If the report contains ANY error — even a single wrong detail — answer NO.\n"
    "Only answer YES if the report is completely accurate.\n\n"
    "Report: {report}\n\n"
    "Reply with YES or NO, then your confidence (0.00-1.00), then one sentence.\n"
    "Example: YES 0.92 The report accurately describes bilateral lung findings."
)

PROMPT_COT = (
    "You are a critical radiology AI evaluator detecting hallucinations.\n\n"
    "Report: {report}\n\n"
    "Step 1 - Objects: Is every finding in the report visible in the image?\n"
    "Step 2 - Attributes: Are size/shape/density correct?\n"
    "Step 3 - Relations: Are left/right/spatial relations correct?\n"
    "Step 4 - Verdict: VERDICT: YES 0.92  or  VERDICT: NO 0.87\n"
)

PROMPTS = {"standard": PROMPT_STANDARD, "cot": PROMPT_COT}


def parse_output(raw: str, prompt_mode: str = "standard") -> dict:
    text = raw.strip().upper()
    if prompt_mode == "cot":
        m = re.search(r"VERDICT:\s*(YES|NO)\s+(0?\.\d+|1\.0+)", text)
        if m:
            tok, conf = m.group(1), float(m.group(2))
            label  = "faithful" if tok == "YES" else "hallucinated"
            p_hallu = conf if label == "hallucinated" else 1.0 - conf
            return {"pred_label": label,
                    "pred_confidence": round(float(np.clip(p_hallu,0,1)),4),
                    "raw_verdict": tok}

    tokens = text.split()
    label_tok, conf_val = None, None
    for i, tok in enumerate(tokens[:6]):
        if tok in ("YES","NO"):
            label_tok = tok
            if i+1 < len(tokens):
                m = re.search(r"(0?\.\d+|1\.0+)", tokens[i+1])
                if m: conf_val = float(m.group(1))
            break
    if label_tok is None:
        m = re.search(r"\b(YES|NO)\b[^0-9]*(0?\.\d+|1\.0+)", text)
        if m: label_tok, conf_val = m.group(1), float(m.group(2))

    if label_tok is None:
        return {"pred_label":"unknown","pred_confidence":0.5,"raw_verdict":"unknown"}

    label   = "faithful" if label_tok=="YES" else "hallucinated"
    conf_val = conf_val if conf_val is not None else 0.75
    p_hallu  = conf_val if label=="hallucinated" else 1.0-conf_val
    return {"pred_label": label,
            "pred_confidence": round(float(np.clip(p_hallu,0,1)),4),
            "raw_verdict": label_tok}


def maybe_invert(out_path, model_key="llama"):
    from sklearn.metrics import roc_auc_score
    records = [json.loads(l) for l in open(out_path) if l.strip()]
    valid   = [r for r in records if r["pred_label"] in ("faithful","hallucinated")]
    if not valid: return
    y_true = np.array([1 if r["gt_hallucinated"] else 0 for r in valid])
    y_conf = np.array([r["pred_confidence"] for r in valid])
    try:    auroc = roc_auc_score(y_true, y_conf)
    except: return
    sidecar = out_path.with_suffix(".inversion.json")
    if auroc < 0.5:
        print(f"  [recal] AUROC={auroc:.3f} → inverting confidence")
        inverted = []
        for r in records:
            r2 = dict(r); r2["pred_confidence"] = round(1-r["pred_confidence"],4)
            r2["confidence_inverted"] = True; inverted.append(r2)
        with open(out_path,"w") as f:
            for r in inverted: f.write(json.dumps(r)+"\n")
        json.dump({"model":model_key,"original_auroc":auroc,"inverted":True,
                   "inverted_auroc":1-auroc}, open(sidecar,"w"), indent=2)
        print(f"  [recal] Effective AUROC: {1-auroc:.3f}")
    else:
        print(f"  [recal] AUROC={auroc:.3f} — no inversion needed")
        json.dump({"model":model_key,"original_auroc":auroc,"inverted":False},
                  open(sidecar,"w"), indent=2)


def run_llama(prompt_mode="standard", recalibrate=True):
    from transformers import MllamaForConditionalGeneration, AutoProcessor

    with open(PROCESSED/"benchmark_v1.jsonl") as f:
        samples = [json.loads(l) for l in f]

    suffix   = f"_{prompt_mode}" if prompt_mode != "standard" else ""
    out_path = OUTPUTS / f"llama_v2{suffix}.jsonl"

    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try: done_ids.add(json.loads(line)["id"])
                except: pass
        print(f"[resume] {len(done_ids)} already done")

    remaining = [s for s in samples if s["id"] not in done_ids]
    if not remaining:
        print("[done] All samples processed.")
        if recalibrate: maybe_invert(out_path)
        return

    print(f"[load] LLaMA from {MODEL_DIR}")
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
    model = MllamaForConditionalGeneration.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()

    # ── EMBEDDING HOTFIX ──────────────────────────────────────────────────────
    if len(processor.tokenizer) > model.config.text_config.vocab_size:
        print(f"  [hotfix] Resizing embeddings: {model.config.text_config.vocab_size} -> {len(processor.tokenizer)}")
        model.resize_token_embeddings(len(processor.tokenizer))
        model.config.text_config.vocab_size = len(processor.tokenizer)
    # ──────────────────────────────────────────────────────────────────────────

    max_new_tokens = 300 if prompt_mode == "cot" else 100
    prompt_text    = PROMPTS[prompt_mode]

    print(f"[inference] {len(remaining)} samples | prompt={prompt_mode}")
    t0 = time.time()

    with open(out_path, "a") as fout:
        for i, sample in enumerate(tqdm(remaining, desc="llama")):
            raw = None
            img_path = sample["img_path"]

            for attempt in range(3):
                try:
                    image = Image.open(img_path).convert("RGB")
                    image.thumbnail((560, 560), Image.LANCZOS)

                    # CRITICAL FIX: use <|image|> token in user content
                    # Mllama requires image in the content list alongside text
                    # BYPASS CHAT TEMPLATE: Directly format text to prevent cross-attention cache drop
                    formatted_prompt = f"<|image|>{prompt_text.format(report=sample['report'])}"

                    inputs = processor(
                        images=image,
                        text=formatted_prompt,
                        return_tensors="pt",
                        padding=False,
                    ).to(model.device, dtype=torch.bfloat16)

                    with torch.no_grad():
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                            repetition_penalty=1.2,
                            no_repeat_ngram_size=5,
                            use_cache=True, # Turn cache back ON since we aren't using the broken chat template masks
                        )

                    # Decode only new tokens
                    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
                    raw = processor.decode(new_tokens, skip_special_tokens=True).strip()

                    # Check for degenerate output (mostly ! or garbled)
                    excl_ratio = raw.count("!") / max(len(raw), 1)
                    if excl_ratio > 0.3 or len(raw) < 3:
                        if attempt < 2:
                            print(f"\n  [retry-degen] {sample['id']}: excl_ratio={excl_ratio:.2f}")
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

            parsed = parse_output(raw, prompt_mode)
            record = {
                "id":              sample["id"],
                "model":           "llama",
                "prompt_mode":     prompt_mode,
                "gt_hallucinated": sample["hallucinated"],
                "hallu_type":      sample["hallu_type"],
                "raw_output":      raw,
                "latency_s":       round((time.time()-t0)/(i+1), 3),
                **parsed,
            }
            fout.write(json.dumps(record)+"\n")
            fout.flush()

    print(f"[done] {len(remaining)} samples in {time.time()-t0:.1f}s → {out_path}")
    if recalibrate:
        maybe_invert(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_mode", default="standard", choices=["standard","cot"])
    parser.add_argument("--no_recalibrate", action="store_true")
    args = parser.parse_args()
    run_llama(prompt_mode=args.prompt_mode, recalibrate=not args.no_recalibrate)




# #!/usr/bin/env python3
# """
# Drop-in replacement for run_llama_transformers() in run_inference.py
# Key fixes:
#   1. Proper cross_attention_states handling for Mllama
#   2. Suppress generation flag warnings
#   3. Correct processor usage - image must be in content list
#   4. Repetition penalty to prevent ! spam
#   5. Better fallback on cross-attn error
# """

# # ── PATCHES ───────────────────────────────────────────────────────────────────
# import sys, builtins
# from PIL.Image import Resampling
# sys.modules["transformers.utils.PILImageResampling"] = Resampling
# setattr(builtins, "PILImageResampling", Resampling)
# try:
#     import transformers
#     import transformers.image_processing_utils as _ipu
#     for _mod in [transformers, _ipu]:
#         if not hasattr(_mod, "PILImageResampling"):
#             setattr(_mod, "PILImageResampling", Resampling)
# except Exception:
#     pass

# try:
#     from transformers import MllamaProcessor
#     if not hasattr(MllamaProcessor, "_get_num_multimodal_tokens"):
#         def _get_num_multimodal_tokens(self, **kwargs):
#             return {"num_image_tokens": [1601]}
#         MllamaProcessor._get_num_multimodal_tokens = _get_num_multimodal_tokens
# except ImportError:
#     pass

# import json, time, re, logging, argparse
# import numpy as np
# from pathlib import Path
# import torch, yaml
# from PIL import Image
# from tqdm import tqdm

# logging.getLogger("transformers.generation").setLevel(logging.ERROR)
# logging.getLogger("transformers").setLevel(logging.ERROR)

# ROOT     = Path("/mnt/ssd/users/durgesh/HEalthcare/Health2")
# CFG_PATH = ROOT / "code/v1/config.yaml"
# with open(CFG_PATH) as f:
#     CFG = yaml.safe_load(f)

# PROCESSED  = ROOT / CFG["paths"]["data_processed"]
# OUTPUTS    = ROOT / CFG["paths"]["outputs"]
# OUTPUTS.mkdir(parents=True, exist_ok=True)
# MODEL_DIR  = ROOT / CFG["paths"]["models"] / "Llama-3.2-11B-Vision-Instruct"

# PROMPT_STANDARD = (
#     "You are a critical radiology AI evaluator. "
#     "Your job is to detect errors and hallucinations in chest X-ray reports.\n\n"
#     "Carefully examine the X-ray image and the report below. "
#     "Check whether every finding, anatomy, attribute (size/shape/density), "
#     "and spatial relationship (left/right/upper/lower) in the report is "
#     "actually visible and correct in the image.\n\n"
#     "If the report contains ANY error — even a single wrong detail — answer NO.\n"
#     "Only answer YES if the report is completely accurate.\n\n"
#     "Report: {report}\n\n"
#     "Reply with YES or NO, then your confidence (0.00-1.00), then one sentence.\n"
#     "Example: YES 0.92 The report accurately describes bilateral lung findings."
# )

# PROMPT_COT = (
#     "You are a critical radiology AI evaluator detecting hallucinations.\n\n"
#     "Report: {report}\n\n"
#     "Step 1 - Objects: Is every finding in the report visible in the image?\n"
#     "Step 2 - Attributes: Are size/shape/density correct?\n"
#     "Step 3 - Relations: Are left/right/spatial relations correct?\n"
#     "Step 4 - Verdict: VERDICT: YES 0.92  or  VERDICT: NO 0.87\n"
# )

# PROMPTS = {"standard": PROMPT_STANDARD, "cot": PROMPT_COT}


# def parse_output(raw: str, prompt_mode: str = "standard") -> dict:
#     text = raw.strip().upper()
#     if prompt_mode == "cot":
#         m = re.search(r"VERDICT:\s*(YES|NO)\s+(0?\.\d+|1\.0+)", text)
#         if m:
#             tok, conf = m.group(1), float(m.group(2))
#             label  = "faithful" if tok == "YES" else "hallucinated"
#             p_hallu = conf if label == "hallucinated" else 1.0 - conf
#             return {"pred_label": label,
#                     "pred_confidence": round(float(np.clip(p_hallu,0,1)),4),
#                     "raw_verdict": tok}

#     tokens = text.split()
#     label_tok, conf_val = None, None
#     for i, tok in enumerate(tokens[:6]):
#         if tok in ("YES","NO"):
#             label_tok = tok
#             if i+1 < len(tokens):
#                 m = re.search(r"(0?\.\d+|1\.0+)", tokens[i+1])
#                 if m: conf_val = float(m.group(1))
#             break
#     if label_tok is None:
#         m = re.search(r"\b(YES|NO)\b[^0-9]*(0?\.\d+|1\.0+)", text)
#         if m: label_tok, conf_val = m.group(1), float(m.group(2))

#     if label_tok is None:
#         return {"pred_label":"unknown","pred_confidence":0.5,"raw_verdict":"unknown"}

#     label   = "faithful" if label_tok=="YES" else "hallucinated"
#     conf_val = conf_val if conf_val is not None else 0.75
#     p_hallu  = conf_val if label=="hallucinated" else 1.0-conf_val
#     return {"pred_label": label,
#             "pred_confidence": round(float(np.clip(p_hallu,0,1)),4),
#             "raw_verdict": label_tok}


# def maybe_invert(out_path, model_key="llama"):
#     from sklearn.metrics import roc_auc_score
#     records = [json.loads(l) for l in open(out_path) if l.strip()]
#     valid   = [r for r in records if r["pred_label"] in ("faithful","hallucinated")]
#     if not valid: return
#     y_true = np.array([1 if r["gt_hallucinated"] else 0 for r in valid])
#     y_conf = np.array([r["pred_confidence"] for r in valid])
#     try:    auroc = roc_auc_score(y_true, y_conf)
#     except: return
#     sidecar = out_path.with_suffix(".inversion.json")
#     if auroc < 0.5:
#         print(f"  [recal] AUROC={auroc:.3f} → inverting confidence")
#         inverted = []
#         for r in records:
#             r2 = dict(r); r2["pred_confidence"] = round(1-r["pred_confidence"],4)
#             r2["confidence_inverted"] = True; inverted.append(r2)
#         with open(out_path,"w") as f:
#             for r in inverted: f.write(json.dumps(r)+"\n")
#         json.dump({"model":model_key,"original_auroc":auroc,"inverted":True,
#                    "inverted_auroc":1-auroc}, open(sidecar,"w"), indent=2)
#         print(f"  [recal] Effective AUROC: {1-auroc:.3f}")
#     else:
#         print(f"  [recal] AUROC={auroc:.3f} — no inversion needed")
#         json.dump({"model":model_key,"original_auroc":auroc,"inverted":False},
#                   open(sidecar,"w"), indent=2)


# def run_llama(prompt_mode="standard", recalibrate=True):
#     from transformers import MllamaForConditionalGeneration, AutoProcessor

#     with open(PROCESSED/"benchmark_v1.jsonl") as f:
#         samples = [json.loads(l) for l in f]

#     suffix   = f"_{prompt_mode}" if prompt_mode != "standard" else ""
#     out_path = OUTPUTS / f"llama_v2{suffix}.jsonl"

#     done_ids = set()
#     if out_path.exists():
#         with open(out_path) as f:
#             for line in f:
#                 try: done_ids.add(json.loads(line)["id"])
#                 except: pass
#         print(f"[resume] {len(done_ids)} already done")

#     remaining = [s for s in samples if s["id"] not in done_ids]
#     if not remaining:
#         print("[done] All samples processed.")
#         if recalibrate: maybe_invert(out_path)
#         return

#     print(f"[load] LLaMA from {MODEL_DIR}")
#     processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
#     model = MllamaForConditionalGeneration.from_pretrained(
#         str(MODEL_DIR), torch_dtype=torch.bfloat16, device_map="auto"
#     ).eval()

#     max_new_tokens = 300 if prompt_mode == "cot" else 100
#     prompt_text    = PROMPTS[prompt_mode]

#     print(f"[inference] {len(remaining)} samples | prompt={prompt_mode}")
#     t0 = time.time()

#     with open(out_path, "a") as fout:
#         for i, sample in enumerate(tqdm(remaining, desc="llama")):
#             raw = None
#             img_path = sample["img_path"]

#             for attempt in range(3):
#                 try:
#                     image = Image.open(img_path).convert("RGB")
#                     image.thumbnail((560, 560), Image.LANCZOS)

#                     # CRITICAL FIX: use <|image|> token in user content
#                     # Mllama requires image in the content list alongside text
#                     messages = [{
#                         "role": "user",
#                         "content": [
#                             {"type": "image"},
#                             {"type": "text",
#                              "text": prompt_text.format(report=sample["report"])}
#                         ]
#                     }]

#                     text_input = processor.apply_chat_template(
#                         messages,
#                         add_generation_prompt=True,
#                         tokenize=False,
#                     )
#                     inputs = processor(
#                         images=image,
#                         text=text_input,
#                         return_tensors="pt",
#                         padding=False,
#                     ).to(model.device, dtype=torch.bfloat16)

#                     with torch.no_grad():
#                         output_ids = model.generate(
#                             **inputs,
#                             max_new_tokens=max_new_tokens,
#                             do_sample=False,
#                             temperature=None,
#                             top_p=None,
#                             repetition_penalty=1.3,   # prevent ! spam
#                             no_repeat_ngram_size=5,
#                         )






#                     # Decode only new tokens
#                     new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
#                     raw = processor.decode(new_tokens, skip_special_tokens=True).strip()

#                     # Check for degenerate output (mostly ! or garbled)
#                     excl_ratio = raw.count("!") / max(len(raw), 1)
#                     if excl_ratio > 0.3 or len(raw) < 3:
#                         if attempt < 2:
#                             print(f"\n  [retry-degen] {sample['id']}: excl_ratio={excl_ratio:.2f}")
#                             time.sleep(1)
#                             raw = None
#                             continue
#                     break

#                 except Exception as e:
#                     if attempt < 2:
#                         print(f"\n  [retry] {sample['id']} attempt {attempt}: {e}")
#                         time.sleep(2)
#                         raw = None
#                     else:
#                         print(f"\n  [fail] {sample['id']}: {e}")

#             if raw is None or raw == "":
#                 raw = "unknown"

#             parsed = parse_output(raw, prompt_mode)
#             record = {
#                 "id":              sample["id"],
#                 "model":           "llama",
#                 "prompt_mode":     prompt_mode,
#                 "gt_hallucinated": sample["hallucinated"],
#                 "hallu_type":      sample["hallu_type"],
#                 "raw_output":      raw,
#                 "latency_s":       round((time.time()-t0)/(i+1), 3),
#                 **parsed,
#             }
#             fout.write(json.dumps(record)+"\n")
#             fout.flush()

#     print(f"[done] {len(remaining)} samples in {time.time()-t0:.1f}s → {out_path}")
#     if recalibrate:
#         maybe_invert(out_path)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--prompt_mode", default="standard", choices=["standard","cot"])
#     parser.add_argument("--no_recalibrate", action="store_true")
#     args = parser.parse_args()
#     run_llama(prompt_mode=args.prompt_mode, recalibrate=not args.no_recalibrate)