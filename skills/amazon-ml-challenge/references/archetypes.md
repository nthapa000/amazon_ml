# Problem archetypes and how to stack levers

Identify which archetype the new problem resembles (it may mix two). Stack levers in the listed
order; each later lever assumes the earlier ones are in place. Keep a lever only if OOF improves.

## Archetype 1: regression from text + image, relative metric (2025-like)

```
parse text ─► tab feats (qty, pack, unit, brand TE, LLM fields) ─┐
text  ─► fine-tuned numeric-aware encoder ────────────────────────┤
image ─► SigLIP / CLIP-L (fine-tune last blocks) ─────────────────┼─► cross-attn fusion ─► small FFN ─► log y
OOF kNN neighbour target stats (FAISS) ───────────────────────────┘
loss: L1(log) warm-up → + SMAPE(log); aux InfoNCE; EMA; 3 seeds
ensemble (log space): GBDT on [tab + frozen emb + kNN], per-unit-target model
post: OOF multiplier ─► clip ─► snap if targets repeat
```
Order of expected gain: log target + L1 → fine-tuned text → add image → parsed quantity features →
kNN features → fusion upgrade → SMAPE loss → ensemble → calibration.
Scripts: parse_catalog → knn_feats → gbdt_cv (lgb, xgb, cat) → mm_finetune → blend → sanity.

## Archetype 2: structured extraction from images (2024-like)

```
image ─► resize ≤ model max (keep aspect) ─► VLM zero-shot, strict format prompt
       └► OCR text appended to prompt for small VLMs
errors on val ─► bucket by entity / unit ─► few-shot exemplar pool
curated 1-2K gold rows ─► QLoRA SFT (+ rotation aug)
vote(SFT, few-shot, zero-shot model B) ─► regex normaliser ─► allowed-unit filter ─► abstain if low confidence
```
Validation: hold out 2K labelled rows stratified by entity type; score with the exact official F1.
Scripts: download_images, sanity (fmt=value_unit units=...). VLM code: LLaMA-Factory or Unsloth notebooks.

## Archetype 3: regression from text + category id, huge data (2023-like)

```
title+bullets+desc ─► BERT / RoBERTa fine-tune + learned category embedding ─► log y (clipped, z-scored)
GBDT on TF-IDF SVD + text stats + category target encoding
combine by metric-aware rule (MAPE → shrink / min) ─► snap to train-value grid
```

## Archetype 4: extreme multi-class classification (2021-like)

```
TF-IDF (word + char) ─► LinearSVC / SGD          (fast, strong baseline, ALL rows)
multilingual transformer fine-tune on ALL rows (streaming loader)
sentence embeddings ─► GPU kNN vote
blend class probabilities (rank-normalised) ─► argmax ; prior correction if test drifts
```

## Archetype 5: tabular classification (scientist's churn demo)

```
eda.py ─► drop ids + redundant cols, 0/1 binaries, one-hot / TE ─► gbdt_cv.py (lgb/xgb/cat, NaN native)
─► Optuna trials ─► blend.py (tunes threshold for F1 on OOF) ─► sanity.py
```
If Amazon provides SageMaker: same data, CSV target-first/no-header, AMT for tuning (references/sagemaker.md).

## Synergy matrix

| Combine | With | Why stronger together |
|---|---|---|
| Log target + L1 | exact SMAPE after warm-up | L1 finds the basin; SMAPE refines where metric saturates |
| Parsed quantity | per-unit alternate target | second, differently biased ensemble member |
| Frozen embeddings | OOF kNN neighbour targets | similarity turns into direct price evidence |
| Error mining | dynamic few-shot exemplars | model sees its own failure modes |
| Curated gold set | QLoRA SFT | clean supervision (0.68 → 0.87 F1) |
| OCR text | small VLM | exact digits the VLM misreads |
| Asymmetric metric (MAPE) | min / down-shifted ensemble | exploits the cheaper error direction |
| Fine-tuned multimodal net | GBDT on tab + embeddings | low error correlation |
| Adversarial validation | duplicate-grouped folds | CV that tracks the private LB |
| Contrastive aux loss | small FFN head | regularised representation resists overfit |
| Probabilistic classifier | OOF threshold tuning | converts good ranking into good F1 |

## EDA signal → first lever

| If EDA shows | Pull first |
|---|---|
| Target spans orders of magnitude | log target, L1/Huber, geometric blending |
| Many identical target values | snap-to-grid; classification-style aux head |
| Numbers and units in text drive the target | regex + LLM extraction, numeric feats into head |
| Label errors in worst OOF rows | curated gold subset, train with/without fixes |
| Near-duplicate listings train↔test | kNN neighbour features; group folds |
| Images contain printed text | higher-res vision encoder, OCR, VLM |
| Mixed-language text | multilingual encoder |
| Train/test drift | adversarial validation; test-like validation rows |
| Exact-match output format | grammar post-processor, allowed-list filter, abstention |
| Imbalanced binary target | class weights, stratified folds, threshold tuning |
| Missing values | native NaN in GBDT + is_missing flags; never drop test rows |
