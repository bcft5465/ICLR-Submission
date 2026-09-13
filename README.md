# SAFER Benchmark — Reproduction Guide

This README takes you from zero to reproduced figures/tables.

## Repository layout

After cloning, your directory must look exactly like this — every script computes `ROOT = Path(__file__).resolve().parents[3]` and will fail silently if the nesting is wrong:

```
<repo_root>/
  code/v1/
    config.yaml
    run_pipeline.sh
    requirements.txt
    01_data_prep/
      build_benchmark_multi.py
      build_benchmark.py
      extract_examples.py
    02_inference/
      run_inference.py
    03_metrics/
      compute_metrics.py
    04_analysis/
      generate_figures.py
  data/v1/            <- you create this; filled by steps below
  results/v1/         <- created automatically when scripts run
```

---

## Step 0 — Python environment

Tested on: **Python 3.11, NVIDIA GPU (sm_120), CUDA 12.8**.

Install order is strict — torch must be pinned before vLLM:

```bash
# Step 1 — PyTorch with CUDA 12.8 (Blackwell sm_120 support)
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Step 2 — vLLM (must match torch==2.10.0 exactly)
pip install vllm==0.19.0

# Step 3 — everything else
pip install -r requirements.txt
```

Verify the install:

```bash
python -c "
import torch, vllm, transformers, open_clip
print('torch:', torch.__version__)          # 2.10.0+cu128
print('cuda:', torch.cuda.is_available())   # True
print('gpu:', torch.cuda.get_device_name(0))
print('vllm:', vllm.__version__)            # 0.19.0
print('transformers:', transformers.__version__)  # 4.56.0
print('open_clip: ok')
"
```

> **Critical constraints:**
>
> - Do **not** upgrade torch past 2.10.0 — vLLM 0.19.0 requires exactly 2.10.0
> - Do **not** install `open-flamingo` — it downgrades torch to 2.0.1 and breaks the environment
> - If you see `CUDA error: no kernel image is available`, your torch does not support sm_120 — reinstall with the cu128 index URL above

---

## Step 1 — Download datasets

Create the data directories first:

```bash
mkdir -p data/v1/raw/images/images_normalized
mkdir -p data/v1/raw/mimiccxr
mkdir -p data/v1/processed
mkdir -p data/v1/models
```

### 1a. OpenI (IU X-Ray) — free, no account needed

```bash
cd data/v1/raw

# Images (~1.2 GB)
wget https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz
tar -xzf NLMCXR_png.tgz -C images/
# Images land in images/images_normalized/*.png

# Reports (small)
wget https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz
tar -xzf NLMCXR_reports.tgz
# Produces indiana_reports.csv and indiana_projections.csv in data/v1/raw/

cd ../../..
```

Verify:

```bash
ls data/v1/raw/indiana_reports.csv data/v1/raw/indiana_projections.csv
ls data/v1/raw/images/images_normalized/ | wc -l   # ~3,955 files
```

### 1b. MIMIC-CXR (Kaggle mirror) — requires free Kaggle account

1. Create an account at https://www.kaggle.com
2. Go to https://www.kaggle.com/settings → API → "Create New Token" → saves `kaggle.json`
3. Place it: `mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`

```bash
kaggle datasets download -d simhadrisadaram/mimic-cxr-dataset \
    -p data/v1/raw/mimiccxr/ --unzip
```

Expected layout after download:

```
data/v1/raw/mimiccxr/
  mimic_cxr_aug_train.csv
  mimic_cxr_aug_validate.csv
  official_data_iccv_final/
    files/
      p10/  p11/  p12/  p13/  p14/  p15/  p16/  p17/  p18/  p19/
```

Verify:

```bash
head -1 data/v1/raw/mimiccxr/mimic_cxr_aug_train.csv
# Expected: Unnamed: 0.1,Unnamed: 0,subject_id,image,view,AP,PA,Lateral,text,text_augment

ls data/v1/raw/mimiccxr/official_data_iccv_final/files/
# Expected: p10  p11  p12  p13  p14  p15  p16  p17  p18  p19
```

> **Note on missing images:** The Kaggle mirror is incomplete — \~13,560 PA candidates have no image file (evenly spread across p10–p19). This is an upstream issue, not a path error. The pipeline accepts this. Your final benchmark will have \~8,294 samples rather than the theoretical maximum of 9,000.

---

## Step 2 — Download model weights

All models go under `data/v1/models/`. Subdirectory names must match the `local_dir` values in `config.yaml` exactly.

```bash
cd data/v1/models/

# ── vLLM-served generative VLMs ─────────────────────────────────────────────

# Qwen2.5-VL-7B (~15 GB)
hf download Qwen/Qwen2.5-VL-7B-Instruct \
    --local-dir Qwen2.5-VL-7B-Instruct --repo-type model

# MedGemma-27B (~55 GB)
# Accept license first: https://huggingface.co/google/medgemma-27b-it
hf download google/medgemma-27b-it \
    --local-dir medgemma-27b-it --repo-type model

# ── Transformers generative VLMs ────────────────────────────────────────────

# LLaVA-1.5-13B (~27 GB)
hf download llava-hf/llava-1.5-13b-hf \
    --local-dir llava-1.5-13b-hf --repo-type model

# Phi-3-Vision-4B (~8 GB)
hf download microsoft/Phi-3-vision-128k-instruct \
    --local-dir Phi-3-vision-128k-instruct --repo-type model

# CheXagent-8B (~16 GB)
hf download StanfordAIMI/CheXagent-8b \
    --local-dir CheXagent-8b --repo-type model

# ── Embedding / similarity models ───────────────────────────────────────────

# BioViL-T (< 1 GB)
hf download microsoft/BiomedVLP-BioViL-T \
    --local-dir BiomedVLP-BioViL-T --repo-type model

# BiomedCLIP-300M (< 1 GB)
hf download microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
    --local-dir BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 --repo-type model

# BLIP-2-OPT-6.7B (~14 GB)
hf download Salesforce/blip2-opt-6.7b \
    --local-dir blip2-opt-6.7b --repo-type model

cd ../../..
```

**Model summary (\~135 GB total):**

| Config key | HuggingFace repo | Size | Backend |
| --- | --- | --- | --- |
| `qwen25_vl` | `Qwen/Qwen2.5-VL-7B-Instruct` | 7B | vLLM |
| `medgemma` | `google/medgemma-27b-it` | 27B | vLLM |
| `llava15` | `llava-hf/llava-1.5-13b-hf` | 13B | transformers |
| `phi3_vision` | `microsoft/Phi-3-vision-128k-instruct` | 4B | transformers |
| `chexagent` | `StanfordAIMI/CheXagent-8b` | 8B | transformers |
| `biovil_t` | `microsoft/BiomedVLP-BioViL-T` | — | transformers (embed) |
| `biomedclip` | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | 300M | open_clip (embed) |
| `blip2` | `Salesforce/blip2-opt-6.7b` | 7B | transformers (embed) |

**Models NOT supported (do not download):**

- `llama` (LLaMA-3.2-11B-Vision) — cross-attention incompatibility with transformers 4.56.0
- `llava_med` (microsoft/LLaVA-Med) — not publicly available on HuggingFace
- `med_flamingo` — requires `open-flamingo` which downgrades torch to 2.0.1

Verify all weights:

```bash
ls data/v1/models/
# Expected 8 directories:
# BiomedCLIP-PubMedBERT_256-vit_base_patch16_224  BiomedVLP-BioViL-T
# blip2-opt-6.7b  CheXagent-8b  llava-1.5-13b-hf
# medgemma-27b-it  Phi-3-vision-128k-instruct  Qwen2.5-VL-7B-Instruct
```

---

## Step 3 — Verify full layout before running

```
<repo_root>/
  code/v1/
    config.yaml  run_pipeline.sh  requirements.txt
    01_data_prep/  02_inference/  03_metrics/  04_analysis/
  data/v1/
    raw/
      indiana_reports.csv
      indiana_projections.csv
      images/images_normalized/   (~3,955 .png files)
      mimiccxr/
        mimic_cxr_aug_train.csv
        mimic_cxr_aug_validate.csv
        official_data_iccv_final/files/p10..p19/
    models/
      Qwen2.5-VL-7B-Instruct/
      medgemma-27b-it/
      llava-1.5-13b-hf/
      Phi-3-vision-128k-instruct/
      CheXagent-8b/
      BiomedVLP-BioViL-T/
      BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/
      blip2-opt-6.7b/
    processed/   <- populated by Stage 1
```

---

## Option A — Run full pipeline (recommended)

```bash
cd code/v1
bash run_pipeline.sh all
```

This runs all 4 stages in order (\~24–48 hours total on 2 GPUs).

Other modes:

```bash
bash run_pipeline.sh baselines          # baselines only (< 5 min, CPU)
bash run_pipeline.sh medgemma           # single model only
bash run_pipeline.sh all --legacy-single-source   # reproduce original 500-sample submission
```

---

## Option B — Run each stage manually

### Stage 1 — Build the benchmark

```bash
cd code/v1/01_data_prep

python build_benchmark_multi.py

python extract_examples.py
```

Expected output:

```
[config] sources=['openi', 'mimiccxr']  n_total=None  per_source_cap=4500  injection_rate=0.5
[load] openi ...
  → openi: using all 3794 available
[load] mimiccxr ...
[mimiccxr] 31866 valid PA-view records loaded
  → mimiccxr: capped to 4500
[combine] 8294 total records
[done] 8294 samples → data/v1/processed/benchmark_multi_v1.jsonl
  hallucinated: 3903  clean: 4391
  type breakdown: {'object': 1325, 'attribute': 1341, 'relational': 1237}
  source breakdown: {'openi': 3794, 'mimiccxr': 4500}
```

---

### Stage 2 — Run inference

All runs are **resumable** — re-running skips already-written samples.

**2× GPU setup (96 GB each):**

```bash
# GPU 0 — large models
tmux new -s medgemma  && CUDA_VISIBLE_DEVICES=0 python run_inference.py --model medgemma
tmux new -s llava15   && CUDA_VISIBLE_DEVICES=0 python run_inference.py --model llava15
tmux new -s chexagent && CUDA_VISIBLE_DEVICES=0 python run_inference.py --model chexagent

# GPU 1 — smaller models (run in parallel)
tmux new -s qwen       && CUDA_VISIBLE_DEVICES=1 python run_inference.py --model qwen25_vl
tmux new -s phi3       && CUDA_VISIBLE_DEVICES=1 python run_inference.py --model phi3_vision
tmux new -s blip2      && CUDA_VISIBLE_DEVICES=1 python run_inference.py --model blip2
tmux new -s biomedclip && CUDA_VISIBLE_DEVICES=1 python run_inference.py --model biomedclip
tmux new -s biovil     && CUDA_VISIBLE_DEVICES=1 python run_inference.py --model biovil_t

# CPU — baselines (no GPU, < 1 min)
python run_inference.py --model baselines
```

Detach with `Ctrl+B, D`; reattach with `tmux attach -t <name>`.

**Single GPU (sequential):**

```bash
CUDA_VISIBLE_DEVICES=0 python run_inference.py --model all
python run_inference.py --model baselines
```

**CoT ablation (optional):**

```bash
CUDA_VISIBLE_DEVICES=0 python run_inference.py --model medgemma --prompt_mode cot
CUDA_VISIBLE_DEVICES=0 python run_inference.py --model llava15   --prompt_mode cot
CUDA_VISIBLE_DEVICES=1 python run_inference.py --model qwen25_vl --prompt_mode cot
CUDA_VISIBLE_DEVICES=1 python run_inference.py --model phi3_vision --prompt_mode cot
```

**Approximate runtimes (8,294 samples, single GPU):**

| Model | Runtime |
| --- | --- |
| `medgemma` | 6–10 hours |
| `llava15` | 10–12 hours |
| `chexagent` | 3–4 hours |
| `qwen25_vl` | 2–4 hours |
| `blip2` | 2–3 hours |
| `phi3_vision` | 3–5 hours |
| `biovil_t` | 1–2 hours |
| `biomedclip` | 1–2 hours |
| baselines | &lt; 1 minute |

**Monitor progress:**

```bash
watch -n 30 "
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
echo '---'
for m in qwen25_vl medgemma biovil_t chexagent biomedclip phi3_vision blip2 llava15; do
    f=results/v1/outputs/\${m}_v3.jsonl
    count=\$(wc -l < \$f 2>/dev/null || echo 0)
    pct=\$(echo \"scale=1; \$count * 100 / 8294\" | bc 2>/dev/null || echo '?')
    echo \"\$m: \$count / 8294 (\$pct%)\"
done"
```

**Output files produced** (under `results/v1/outputs/`):

- `{model}_v3.jsonl` for each of the 8 models
- `constant_faithful_v3.jsonl`, `random_v3.jsonl`, `text_only_baseline_v3.jsonl`
- `shuffled_image_baseline_qwen25_vl_v3.jsonl`, `shuffled_image_baseline_medgemma_v3.jsonl`

---

### Stage 3 — Compute metrics

```bash
cd code/v1/03_metrics
python compute_metrics.py --model all
```

Safe to run on partial results. Produces under `results/v1/metrics/`:

- `{model}_metrics.json`, `summary.csv`, `halluscore_ablation.csv`
- `significance_tests.json`, `cross_model_verification.json`
- `halluscore_weight_justification.json`, `per_sample.jsonl`

---

### Stage 4 — Generate figures

```bash
cd code/v1/04_analysis
python generate_figures.py
```

Produces under `results/v1/figures/` (PNG + PDF):

- `fig1_halluscore_bar`
- `fig2_roc_curves`
- `fig3_calibration_part{1-4}`
- `fig4_confusion_matrices_part{1-4}`
- `fig5_confidence_dist_part{1-4}`
- `fig6_shuffled_image_comparison`
- `fig7_per_type_accuracy_{object,attribute,relational}`

---

## What you get

```
results/v1/
  examples/examples.md + *.png        <- Section 3 examples
  outputs/*_v3.jsonl                   <- raw predictions
  metrics/summary.csv                  <- Table 1 source
  metrics/halluscore_ablation.csv      <- weight ablation
  metrics/*_metrics.json               <- per-model metrics
  metrics/significance_tests.json      <- McNemar p-values
  figures/fig1..fig7.{png,pdf}         <- all paper figures
```

---

## Key numbers (verbatim for paper)

| Metric | Value |
| --- | --- |
| Sources | OpenI (IU X-Ray), MIMIC-CXR |
| Per-source cap | 4,500 |
| OpenI records used | 3,794 (below cap — not padded) |
| MIMIC-CXR records used | 4,500 (capped from 31,866) |
| Combined benchmark size | 8,294 |
| Hallucinated samples | 3,903 |
| Clean samples | 4,391 |
| Effective hallucination rate | 47.1% |
| Injection failures (no vocab match) | 244 (2.9%) |
| Type breakdown | object 1,325 / attr 1,341 / rel 1,237 |
| Models evaluated | 8 (5 generative VLMs + 3 embed/sim) |
| Shuffled-image baseline models | qwen25_vl, medgemma |

---

## Known issues

**MIMIC-CXR missing images (\~13,560):** Evenly spread across p10–p19 in the Kaggle mirror — incomplete upstream upload, not a path error. Document in paper Limitations.

**Effective injection rate 47.1% vs configured 50%:** 244 reports contain no SWAP_MAP-matchable terms. Report effective rate only.

**CheXagent scoring:** Uses token-F1 between its generated report and the benchmark report (report-comparison mode), not generative prompt-following. Note this methodological difference in Section 4.

**BioViL-T, BiomedCLIP, BLIP-2:** Image-text similarity scoring, not VERDICT/P_ERROR prompt-following. Same note applies.

**MedGemma shuffled-image result:** MedGemma scores higher with a shuffled image than real image (HalluScore 0.6895 vs 0.6758). Text-only baseline also matches MedGemma (0.6778). Strong evidence MedGemma ignores visual input for this task. State explicitly in paper.

**LLaVA-1.5 vs Phi-3-Vision:** Statistically indistinguishable (McNemar p=1.0). Do not claim one outperforms the other.

---

## Troubleshooting

`CUDA error: no kernel image is available`

```bash
pip install torch==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

`ImportError: cannot import name 'infer_schema' from 'torch.library'`torch was upgraded past 2.10.0. Reinstall torch 2.10.0 as above, then:

```bash
pip install vllm==0.19.0
```

`yaml.parser.ParserError` **on startup**Indentation error in `config.yaml`. All model blocks under `models:` must use exactly 2-space indentation.

`DynamicCache has no attribute 'seen_tokens'` **(Phi-3-Vision**)Already patched inside `run_inference.py`. Confirm you are using the latest version from this repo.

`AssertionError: Failed to apply prompt replacement` **(MedGemma + vLLM**)MedGemma requires the exact prompt format: `<bos><start_of_turn>user\n<start_of_image><end_of_image>{prompt}<end_of_turn>\n<start_of_turn>model\n`Do not modify `make_prompt_medgemma` in `run_inference.py`.

`Repository Not Found` **when downloading MedGemma**You must accept the license at https://huggingface.co/google/medgemma-27b-it and be logged in: `huggingface-cli login`