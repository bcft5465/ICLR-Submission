# SAFER Failure-Analysis Experiments — Summary Report

Auto-generated from failure_analysis.py. Each section below reflects what was actually computed from your run's real outputs — sections for experiments that were skipped (missing input files) say so explicitly rather than being silently omitted.

## Experiment 1 — Discrimination-metric degenerate-classifier blind spot

```json
{
  "class_prior_hallucinated": 0.4706,
  "specificity_fixed_at": 0.03,
  "mcc_floor_tested": 0.05,
  "highest_recall_where_original_disc_below_0.10": null,
  "note": "If the ORIGINAL discrimination formula only drops below 0.10 at very high recall (close to 1.0), that IS the blind spot: a model predicting 'hallucinated' ~90-97% of the time (like CheXagent, recall=0.979) still scores well above the near-degenerate threshold. The MCC-gated fix should show a sharper, earlier cutoff.",
  "real_model_discrimination_at_reported_recall": [
    {
      "model": "chexagent",
      "recall": 0.979,
      "specificity": 0.0317,
      "mcc": 0.033,
      "f1": 0.6381,
      "discrimination": 0.5741
    },
    {
      "model": "qwen25_vl",
      "recall": 0.9992,
      "specificity": 0.0057,
      "mcc": 0.0424,
      "f1": 0.641,
      "discrimination": 0.578
    },
    {
      "model": "llava15",
      "recall": 1.0,
      "specificity": 0.0,
      "mcc": 0.0,
      "f1": 0.64,
      "discrimination": 0.5657
    },
    {
      "model": "phi3_vision",
      "recall": 1.0,
      "specificity": 0.0002,
      "mcc": 0.0104,
      "f1": 0.64,
      "discrimination": 0.5686
    },
    {
      "model": "medgemma",
      "recall": 0.5826,
      "specificity": 0.8012,
      "mcc": 0.3948,
      "f1": 0.6451,
      "discrimination": 0.6707
    },
    {
      "model": "biovil_t",
      "recall": 0.0,
      "specificity": 1.0,
      "mcc": 0.0,
      "f1": 0.0,
      "discrimination": 0.0
    },
    {
      "model": "biomedclip",
      "recall": 0.0151,
      "specificity": 0.9565,
      "mcc": -0.0829,
      "f1": 0.0284,
      "discrimination": 0.1141
    },
    {
      "model": "blip2",
      "recall": 0.3356,
      "specificity": 0.6454,
      "mcc": -0.0199,
      "f1": 0.387,
      "discrimination": 0.4355
    },
    {
      "model": "constant_faithful",
      "recall": 0.0,
      "specificity": 1.0,
      "mcc": 0.0,
      "f1": 0.0,
      "discrimination": 0.0
    }
  ]
}
```

## Experiment 2 — MedGemma-vs-cluster disagreement / shortcut hypothesis

```json
{
  "n_common_samples": 8166,
  "cluster_models": [
    "qwen25_vl",
    "llava15",
    "phi3_vision",
    "chexagent"
  ],
  "overall_medgemma_cluster_agreement": 0.3766,
  "report_length_shortcut_check": {
    "agree_mean_words": 48.55642276422764,
    "disagree_mean_words": 42.21921037124337,
    "interpretation": "If disagree_mean_words differs substantially from agree_mean_words, report length may partly explain WHERE medgemma and the cluster diverge (not necessarily WHY either is right)."
  },
  "length_vs_cluster_verdict": {
    "spearman_rho": null,
    "p_value": null,
    "interpretation": "A significant positive rho here would support the 'longer report -> cluster more likely to call it hallucinated' surface-shortcut hypothesis, independent of ground truth."
  },
  "ground_truth_tracking_check": {
    "medgemma_pred_hallucinated_rate": {
      "gt_hallucinated": 0.5806705081194342,
      "gt_faithful": 0.19756209751609935
    },
    "cluster_majority_pred_hallucinated_rate": {
      "gt_hallucinated": 1.0,
      "gt_faithful": 0.999770009199632
    }
  },
  "interpretation_gt_tracking": "Compare the GAP between gt_hallucinated_rate and gt_faithful_rate for medgemma vs. the cluster. A LARGER gap means that model's 'hallucinated' calls track ground truth more; a small gap (rates similar regardless of GT) is consistent with the cluster following a fixed bias/shortcut rather than genuine detection.",
  "disagreement_rate_by_hallu_type": {
    "attribute": 0.3627,
    "object": 0.2069,
    "relational": 0.7086,
    "NaN": 0.8027
  }
}
```

## Experiment 3 — Per-source (OpenI vs MIMIC-CXR) generalization

```json
{
  "benchmark_source_distribution": {
    "openi": 3666,
    "mimiccxr": 4500
  },
  "per_source_metrics": [
    {
      "model": "medgemma",
      "source": "openi",
      "n": 3794,
      "accuracy": 0.669,
      "balanced_acc": 0.6681,
      "f1": 0.6523,
      "mcc": 0.3364,
      "auroc": 0.6881
    },
    {
      "model": "medgemma",
      "source": "mimiccxr",
      "n": 4500,
      "accuracy": 0.7231,
      "balanced_acc": 0.7096,
      "f1": 0.6376,
      "mcc": 0.4559,
      "auroc": 0.7089
    },
    {
      "model": "qwen25_vl",
      "source": "openi",
      "n": 3794,
      "accuracy": 0.4826,
      "balanced_acc": 0.5032,
      "f1": 0.649,
      "mcc": 0.0456,
      "auroc": 0.5156
    },
    {
      "model": "qwen25_vl",
      "source": "mimiccxr",
      "n": 4500,
      "accuracy": 0.4653,
      "balanced_acc": 0.5019,
      "f1": 0.6341,
      "mcc": 0.0416,
      "auroc": 0.5092
    },
    {
      "model": "chexagent",
      "source": "openi",
      "n": 3794,
      "accuracy": 0.4871,
      "balanced_acc": 0.5073,
      "f1": 0.6495,
      "mcc": 0.0582,
      "auroc": 0.5213
    },
    {
      "model": "chexagent",
      "source": "mimiccxr",
      "n": 4500,
      "accuracy": 0.4693,
      "balanced_acc": 0.5034,
      "f1": 0.6283,
      "mcc": 0.0182,
      "auroc": 0.4758
    },
    {
      "model": "llava15",
      "source": "openi",
      "n": 3794,
      "accuracy": 0.4792,
      "balanced_acc": 0.5,
      "f1": 0.6479,
      "mcc": 0.0,
      "auroc": 0.5555
    },
    {
      "model": "llava15",
      "source": "mimiccxr",
      "n": 4500,
      "accuracy": 0.4633,
      "balanced_acc": 0.5,
      "f1": 0.6333,
      "mcc": 0.0,
      "auroc": 0.5506
    },
    {
      "model": "phi3_vision",
      "source": "openi",
      "n": 3794,
      "accuracy": 0.4794,
      "balanced_acc": 0.5003,
      "f1": 0.648,
      "mcc": 0.0156,
      "auroc": 0.5178
    },
    {
      "model": "phi3_vision",
      "source": "mimiccxr",
      "n": 4500,
      "accuracy": 0.4633,
      "balanced_acc": 0.5,
      "f1": 0.6333,
      "mcc": 0.0,
      "auroc": 0.4917
    },
    {
      "model": "biovil_t",
      "source": "openi",
      "n": 3759,
      "accuracy": 0.5204,
      "balanced_acc": 0.5,
      "f1": 0.0,
      "mcc": 0.0,
      "auroc": 0.4778
    },
    {
      "model": "biovil_t",
      "source": "mimiccxr",
      "n": 4500,
      "accuracy": 0.5367,
      "balanced_acc": 0.5,
      "f1": 0.0,
      "mcc": 0.0,
      "auroc": 0.4985
    },
    {
      "model": "biomedclip",
      "source": "openi",
      "n": 3794,
      "accuracy": 0.52,
      "balanced_acc": 0.4999,
      "f1": 0.0319,
      "mcc": -0.0008,
      "auroc": 0.5729
    },
    {
      "model": "biomedclip",
      "source": "mimiccxr",
      "n": 4500,
      "accuracy": 0.508,
      "balanced_acc": 0.4742,
      "f1": 0.0255,
      "mcc": -0.1287,
      "auroc": 0.5074
    },
    {
      "model": "blip2",
      "source": "openi",
      "n": 3794,
      "accuracy": 0.5203,
      "balanced_acc": 0.5187,
      "f1": 0.4893,
      "mcc": 0.0374,
      "auroc": 0.5129
    },
    {
      "model": "blip2",
      "source": "mimiccxr",
      "n": 4500,
      "accuracy": 0.4822,
      "balanced_acc": 0.4636,
      "f1": 0.2732,
      "mcc": -0.0839,
      "auroc": 0.4758
    }
  ],
  "generalization_gap_ranked": [
    {
      "model": "blip2",
      "balanced_acc_gap_across_sources": 0.0551,
      "per_source": {
        "openi": 0.5187,
        "mimiccxr": 0.4636
      }
    },
    {
      "model": "medgemma",
      "balanced_acc_gap_across_sources": 0.0415,
      "per_source": {
        "openi": 0.6681,
        "mimiccxr": 0.7096
      }
    },
    {
      "model": "biomedclip",
      "balanced_acc_gap_across_sources": 0.0257,
      "per_source": {
        "openi": 0.4999,
        "mimiccxr": 0.4742
      }
    },
    {
      "model": "chexagent",
      "balanced_acc_gap_across_sources": 0.0039,
      "per_source": {
        "openi": 0.5073,
    
```

## Experiment 4 — HalluScore rank-instability driver analysis

```json
{
  "component_driver_analysis": [
    {
      "component": "one_minus_ece",
      "rho_vs_hallu_score": -0.1099,
      "rho_vs_hs_equal": -0.3626,
      "rho_vs_hs_calib_heavy": 0.1264,
      "rho_vs_hs_type_heavy": -0.3736,
      "rho_range_across_schemes": 0.5
    },
    {
      "component": "type_acc_macro",
      "rho_vs_hallu_score": 0.6722,
      "rho_vs_hs_equal": 0.8981,
      "rho_vs_hs_calib_heavy": 0.5455,
      "rho_vs_hs_type_heavy": 0.8926,
      "rho_range_across_schemes": 0.3526
    },
    {
      "component": "discrimination",
      "rho_vs_hallu_score": 0.8666,
      "rho_vs_hs_equal": 0.718,
      "rho_vs_hs_calib_heavy": 0.7235,
      "rho_vs_hs_type_heavy": 0.751,
      "rho_range_across_schemes": 0.1486
    }
  ],
  "most_unstable_component": "one_minus_ece",
  "biggest_rank_movers_main_vs_calib_heavy": {
    "text_only_baseline": 4,
    "blip2": 3,
    "phi3_vision": 1,
    "llava15": 1,
    "qwen25_vl": 1
  },
  "interpretation": "'one_minus_ece' shows the widest range of rank-correlation with the final HalluScore across the 5 tested weight schemes, making it the primary driver of rank instability. Models with the largest rank swaps between the main (0.5/0.3/0.2) and calibration-heavy (0.3/0.5/0.2) schemes are listed above -- these are the models whose leaderboard position is most sensitive to a subjective weighting choice, which is itself evidence AGAINST reporting a single scalar HalluScore as the paper's headline ranking mechanism."
}
```

## Experiment 5 — Text-only leakage (trained classifier + span ablation)

```json
{
  "n_train": 6124,
  "n_test": 2042,
  "trained_classifier": {
    "accuracy": 0.8408,
    "balanced_accuracy": 0.8363,
    "f1": 0.8181,
    "mcc": 0.6829,
    "auroc": 0.9183
  },
  "span_ablated_classifier": {
    "accuracy": 0.5602,
    "balanced_accuracy": 0.5363,
    "auroc": 0.6543
  },
  "drop_from_span_ablation": {
    "acc_drop": 0.2806,
    "bal_acc_drop": 0.3,
    "auroc_drop": 0.264
  },
  "top_lexical_features": {
    "top_tokens_predicting_hallucinated": [
      {
        "token": "abnormal",
        "coef": 10.9376
      },
      {
        "token": "livers",
        "coef": 9.2155
      },
      {
        "token": "parenchymal effusion",
        "coef": 7.124
      },
      {
        "token": "parenchymal",
        "coef": 7.0555
      },
      {
        "token": "livers are",
        "coef": 6.3278
      },
      {
        "token": "liver",
        "coef": 6.3087
      },
      {
        "token": "kidney",
        "coef": 5.2593
      },
      {
        "token": "is abnormal",
        "coef": 4.4519
      },
      {
        "token": "the livers",
        "coef": 4.4185
      },
      {
        "token": "diaphragml",
        "coef": 4.3944
      },
      {
        "token": "no parenchymal",
        "coef": 4.0888
      },
      {
        "token": "kidney size",
        "coef": 4.0028
      },
      {
        "token": "severe",
        "coef": 3.7874
      },
      {
        "token": "chronic cardiopulmonary",
        "coef": 3.5183
      },
      {
        "token": "diaphragml effusion",
        "coef": 3.4677
      },
      {
        "token": "abnormal limits",
        "coef": 3.3208
      },
      {
        "token": "within abnormal",
        "coef": 3.287
      },
      {
        "token": "no chronic",
        "coef": 3.2061
      },
      {
        "token": "liver volumes",
        "coef": 2.976
      },
      {
        "token": "small pleural",
        "coef": 2.9322
      }
    ],
    "top_tokens_predicting_faithful": [
      {
        "token": "lungs",
        "coef": -3.2477
      },
      {
        "token": "findings impression",
        "coef": -2.8712
      },
      {
        "token": "lungs are",
        "coef": -2.7466
      },
      {
        "token": "pleural effusion",
        "coef": -2.5465
      },
      {
        "token": "normal",
        "coef": -2.5255
      },
      {
        "token": "pleural",
        "coef": -1.9161
      },
      {
        "token": "lung",
        "coef": -1.8325
      },
      {
        "token": "large pleural",
        "coef": -1.6775
      },
      {
        "token": "impression",
        "coef": -1.6553
      },
      {
        "token": "the lungs",
        "coef": -1.5983
      },
      {
        "token": "or pleural",
        "coef": -1.4989
      },
      {
        "token": "findings",
        "coef": -1.4431
      },
      {
        "token": "nan",
        "coef": -1.4425
      },
      {
        "token": "size normal",
        "coef": -1.4414
      },
      {
        "token": "heart",
        "coef": -1.4022
      },
      {
        "token": "consolidation pleural",
        "coef": -1.3733
      },
      {
        "token": "enlarged",
        "coef": -1.3354
      },
      {
        "token": "normal lungs",
        "coef": -1.2916
      },
      {
        "token": "within normal",
        "coef": -1.245
      },
      {
        "token": "the right",
        "coef": -1.2428
      }
    ]
  },
  "interpretation": "A trained (not heuristic) text-only classifier reaching well above-chance accuracy/AUROC confirms lexical leakage in the SWAP_MAP-based perturbation strategy. If accuracy collapses sharply after removing the exact injected/original span tokens (drop_from_span_ablation large), the leakage is concentrated in the fixed antonym vocabulary itself and is fixable via paraphrase-preserving perturbation generation. If the drop is small, the classifier is exploiting broader linguistic cues beyond the swap vocabulary, which is a harder problem to fix and shou
```

## Experiment 6 — Paraphrase-repair fix for the Exp 5 leak, re-validated

```json
{
  "before_repair": {
    "accuracy": 0.8408,
    "balanced_accuracy": 0.8363,
    "f1": 0.8181,
    "mcc": 0.6829,
    "auroc": 0.9183
  },
  "after_repair": {
    "accuracy": 0.8369,
    "balanced_accuracy": 0.8321,
    "f1": 0.8128,
    "auroc": 0.9162
  },
  "auroc_reduction": 0.0021,
  "n_rule_repaired": 1356,
  "n_llm_repaired": 0,
  "n_llm_failed": 0,
  "rule_application_counts": {
    "collocation:\\bparenchymal effusion\\b->parenchymal opacity": 425,
    "number_agreement:livers->liver": 644,
    "collocation:\\bdiaphragml\\b->diaphragm": 175,
    "collocation:\\bno parenchymal\\b->no significant parenchymal": 216,
    "collocation:\\bchronic cardiopulmonary\\b->longstanding cardiopulmonary": 72
  },
  "repair_mode": "rule_based",
  "interpretation": "Compares the Exp 5 text-only classifier's AUROC before vs. after repairing the specific grammatical/collocation artifacts Exp 5's own feature analysis surfaced. A meaningful auroc_reduction here is direct evidence the leak was fixable at the injection-generation stage, not an inherent property of the task. If AUROC after repair is still well above chance (>0.6), a further, stronger fix (full LLM paraphrase of injected claims, not just grammar repair -- see --exp6_use_llm) is warranted before this benchmark's cross-model comparisons can be trusted at face value."
}
```

## Experiment 7 — MCC-gated Discrimination fix validated on real models

```json
{
  "weights_used": {
    "discrimination": 0.5,
    "ece": 0.3,
    "type_accuracy": 0.2
  },
  "mcc_floors_tested": [
    0.02,
    0.05,
    0.1,
    0.15
  ],
  "per_floor_results": {
    "0.02": {
      "top_vlm_by_gated_score": "chexagent",
      "top_vlm_gated_score": 0.6991,
      "medgemma_rank_among_vlms": 3,
      "chexagent_rank_among_vlms": 1,
      "chexagent_mcc": 0.033,
      "chexagent_discrimination_after_gate": 0.5741,
      "chexagent_gated_below_floor": false
    },
    "0.05": {
      "top_vlm_by_gated_score": "medgemma",
      "top_vlm_gated_score": 0.6757,
      "medgemma_rank_among_vlms": 1,
      "chexagent_rank_among_vlms": 2,
      "chexagent_mcc": 0.033,
      "chexagent_discrimination_after_gate": 0.0,
      "chexagent_gated_below_floor": true
    },
    "0.1": {
      "top_vlm_by_gated_score": "medgemma",
      "top_vlm_gated_score": 0.6757,
      "medgemma_rank_among_vlms": 1,
      "chexagent_rank_among_vlms": 2,
      "chexagent_mcc": 0.033,
      "chexagent_discrimination_after_gate": 0.0,
      "chexagent_gated_below_floor": true
    },
    "0.15": {
      "top_vlm_by_gated_score": "medgemma",
      "top_vlm_gated_score": 0.6757,
      "medgemma_rank_among_vlms": 1,
      "chexagent_rank_among_vlms": 2,
      "chexagent_mcc": 0.033,
      "chexagent_discrimination_after_gate": 0.0,
      "chexagent_gated_below_floor": true
    }
  },
  "medgemma_rank_across_floors": [
    3,
    1,
    1,
    1
  ],
  "inactive_floors_too_low_to_gate_chexagent": [
    0.02
  ],
  "active_floors_that_gate_chexagent_out": [
    0.05,
    0.1,
    0.15
  ],
  "min_floor_that_activates_fix": 0.05,
  "medgemma_rank_stable_among_active_floors": true,
  "original_top_vlm_by_halluscore": "chexagent",
  "interpretation": "Under the ORIGINAL (ungated) formula, chexagent topped the real-VLM leaderboard despite MCC\u22480.03 (near-zero signal) -- this is the exact blind spot Exp 1's synthetic sweep predicted. 'chexagent_gated_below_floor'=True at a given floor means that floor successfully zeroes out chexagent's Discrimination term, removing it from contention for top VLM. Floors BELOW chexagent's actual MCC (~0.033) are trivially inactive (listed in inactive_floors_too_low_to_gate_chexagent) and are NOT part of the stability claim -- a too-low floor being a no-op is expected, not evidence against the fix. The real question is whether medgemma's rank is stable ACROSS THE FLOORS THAT ARE HIGH ENOUGH TO ACTUALLY GATE chexagent out (medgemma_rank_stable_among_active_floors); if True, the fix works and is not sensitive to the exact value chosen once it clears the minimum threshold needed to matter (min_floor_that_activates_fix)."
}
```

## Experiment 8 — Closed-vocabulary hypothesis test (no API needed)

```json
{
  "swap_map_vocabulary_size": 50,
  "swap_map_vocabulary": [
    "abnormal",
    "acute",
    "airspace",
    "anterior",
    "aorta",
    "atelectasis",
    "bilateral",
    "chronic",
    "clavicle",
    "clear",
    "consolidation",
    "decreased",
    "diaphragm",
    "diffuse",
    "edema",
    "effusion",
    "focal",
    "heart",
    "hilar",
    "increased",
    "interstitial",
    "kidney",
    "large",
    "left",
    "liver",
    "lower",
    "lung",
    "mass",
    "mediastinal",
    "mild",
    "new",
    "nodule",
    "normal",
    "opacified",
    "parenchymal",
    "pleura",
    "pleural",
    "pneumothorax",
    "posterior",
    "rib",
    "right",
    "severe",
    "small",
    "spine",
    "stable",
    "sternum",
    "trachea",
    "unilateral",
    "unremarkable",
    "upper"
  ],
  "baseline": {
    "condition": "baseline_no_exclusion",
    "n_vocab_excluded": 0,
    "n_features_used": 5000,
    "accuracy": 0.8408,
    "balanced_accuracy": 0.8363,
    "auroc": 0.9183
  },
  "swap_vocabulary_excluded": {
    "condition": "swap_vocabulary_excluded",
    "n_vocab_excluded": 50,
    "n_features_used": 5000,
    "accuracy": 0.6308,
    "balanced_accuracy": 0.6269,
    "auroc": 0.6893
  },
  "random_control_same_size": {
    "condition": "random_control_same_size",
    "n_vocab_excluded": 50,
    "n_features_used": 5000,
    "accuracy": 0.8413,
    "balanced_accuracy": 0.8369,
    "auroc": 0.9195
  },
  "auroc_drop_from_excluding_swap_vocab": 0.229,
  "auroc_drop_from_random_control": -0.0012,
  "near_chance_threshold_used": 0.6,
  "swap_vocabulary_explains_the_leak": false,
  "interpretation": "Excluding the full SWAP_MAP vocabulary did NOT collapse AUROC to near-chance (or the random control dropped by a comparable amount), meaning the leak is NOT fully explained by the fixed swap vocabulary alone. Some other signal -- report length, punctuation/formatting differences introduced by the injection process, or co-occurring words not in the swap list -- is also contributing meaningfully. This should be investigated before assuming an LLM-paraphrase fix alone will resolve the leak."
}
```

## Experiment 9 — Eligibility-confound test for the residual leak

```json
{
  "text_basis": "9a/9b computed on SWAP_MAP-stripped text (matches Exp 8's 'swap_vocabulary_excluded' condition, AUROC=0.6893); 9c additionally strips a broader radiology vocabulary list on top of SWAP_MAP.",
  "9a_length_confound": {
    "mean_word_count_hallucinated": 39.218,
    "mean_word_count_faithful": 36.369,
    "length_only_classifier_auroc": 0.5716
  },
  "9b_broader_vocab_confound": {
    "mean_broader_vocab_count_hallucinated": 3.0411,
    "mean_broader_vocab_count_faithful": 2.7985,
    "broader_vocab_count_only_classifier_auroc": 0.5604
  },
  "9a_9b_combined_classifier_auroc": 0.5698,
  "9c_full_classifier_with_swap_and_broader_vocab_excluded": {
    "n_vocab_excluded": 99,
    "accuracy": 0.6357,
    "auroc": 0.6954
  },
  "exp8_reference_swap_only_excluded_auroc": 0.6893,
  "additional_auroc_drop_from_broader_vocab_exclusion": -0.0061,
  "near_chance_threshold_used": 0.6,
  "length_or_broader_vocab_explains_residual": false,
  "interpretation": "Even excluding a BROADER radiology vocabulary on top of SWAP_MAP, AUROC remains well above chance. Combined with the length-only and broader-vocab-count-only classifiers' own AUROCs (9a, 9b) as a diagnostic reference, this points toward a STRUCTURAL confound (report length/complexity itself, independent of specific vocabulary) as the more likely explanation for the residual leak, rather than a vocabulary-breadth issue alone. This is a harder problem: it means reports eligible for injection are systematically different in KIND (not just word choice) from reports that are not eligible, which a paraphrase-based fix targeting individual injected claims will NOT resolve -- the sampling/selection process for WHICH reports receive injections would need to be rebalanced (e.g. matching length/complexity between the clean and hallucinated pools) rather than just rewriting the injected text."
}
```

## Experiment 10 — Source-imbalance leak test

```json
{
  "10a_class_balance_by_source": {
    "mimiccxr": {
      "n": 4500,
      "n_hallucinated": 2085,
      "frac_hallucinated": 0.4633
    },
    "openi": {
      "n": 3666,
      "n_hallucinated": 1733,
      "frac_hallucinated": 0.4727
    }
  },
  "10a_max_hallucinated_fraction_gap_across_sources": 0.0094,
  "10b_source_only_classifier_auroc": 0.4941,
  "10c_within_source_classifier_aurocs": {
    "mimiccxr": {
      "n_train": 3375,
      "n_test": 1125,
      "auroc": 0.6611
    },
    "openi": {
      "n_train": 2749,
      "n_test": 917,
      "auroc": 0.6767
    }
  },
  "10c_mean_within_source_auroc": 0.6689,
  "exp9_pooled_auroc_reference": 0.6954,
  "pooled_vs_mean_within_source_auroc_drop": 0.0265,
  "source_leak_confirmed": false,
  "interpretation": "No strong evidence of a source-level confound: class balance is similar across sources (10a), a source-only classifier gets close to chance AUROC (10b), and within-source AUROC is close to the pooled Exp 9 number (10c). This means the residual leak identified in Exp 8/9 is NOT explained by combining two datasets with different reporting conventions -- it appears to be a genuine, source-independent property of the injection/selection process itself, and remains only partially diagnosed after four elimination rounds (grammar, closed vocabulary, broader vocabulary/length, and now source). This should be reported as an open, honestly-unresolved limitation of the benchmark's synthetic perturbation methodology rather than pursued further with additional cheap heuristic probes."
}
```
