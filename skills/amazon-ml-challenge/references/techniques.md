# Technique catalogue with evidence

Tags: **[Seen: Rank X, Year]** = documented by a ranked Amazon ML Challenge team (self-reported).
**[General]** = standard competition practice, not documented by an Amazon ML team.

## Contents
- A. Measured gains (what actually moved scores)
- B. Data curation and cleaning
- C. Feature engineering
- D. Representations and encoders
- E. Multimodal fusion
- F. VLM / LLM strategies
- G. Validation
- H. Ensembling
- I. Post-processing
- J. Compute and throughput

## A. Measured gains

| Change | Before → After | Metric | Team |
|---|---|---|---|
| Curated 1.6K rows instead of 20K noisy rows for QLoRA SFT | 0.679 → 0.865 | F1 | Rank 1, 2024 |
| Idefics2 FT with custom prompt + rotation aug vs zero-shot | 0.44 → 0.616 | F1 | Rank 45, 2024 |
| QLoRA FT (10K) over zero-shot Qwen2-VL-7B | 0.617 → 0.678 | F1 | Rank 1, 2024 |
| Post-processing on zero-shot MiniCPM | 66.2 → 69.3 | F1x100 | Rank 6, 2024 |
| Dynamic few-shot from error pool vs zero-shot | 66.2 → 70.9 | F1x100 | Rank 6, 2024 |
| Voting SFT + few-shot + zero-shot vs best single | 70.9 → 71.8 | F1x100 | Rank 6, 2024 |
| Fine-tuned DistilBERT vs frozen e5 + LightGBM | 50.70 → 45.43 | SMAPE | Rank 8, 2025 |
| + CLIP-L image branch | 45.43 → 42.23 | SMAPE | Rank 8, 2025 |
| CLIP-L/14-336 and 10 epochs | 42.23 → 41.54 | SMAPE | Rank 8, 2025 |
| + DistilBERT branch + contrastive aux losses, 15 epochs | 41.54 → 40.78 | SMAPE | Rank 8, 2025 |
| Full data + transformer vs 50K-sample LinearSVC | 55.48 → 67.47 | Accuracy | Ranks 141 vs 16, 2021 |

Reported 2025 winner: about 39.7 SMAPE. Ladder: frozen emb + GBDT ~50, fine-tuned text ~45,
+ image ~41, fusion + aux + metric loss + ensemble ~40.

## B. Data curation

- Hand-curate a 1K to 2K gold set and fine-tune on it; LoRA memorises label noise. [Seen: Rank 1, 2024]
- Fix invalid labels (invalid units → NA, ranges → rule). [Seen: Ranks 1, 6, 45, 2024]
- Clip extreme log targets (Rank 2 clipped log target at 12). [Seen: Rank 2, 2023]
- De-duplicate; handle missing values. [Seen: Rank 16, 2021]
- Use ALL rows; the 2021 winners credited using the entire training set. [Seen: 2021 winners]
- Worst-OOF review loop: sort OOF by loss, inspect 200 rows, classify label error / parse error / hard,
  fix systematic ones with rules, keep a flag column, re-review after each big model change. [General]

## C. Feature engineering

C1. Parse semi-structured text (scripts/parse_catalog.py): Value/Unit lines, IPQ, "Pack of N",
    "N x M oz", unit normalisation to g / ml / count / cm, brand = first title tokens. [Seen: every 2024 top
    solution parsed units; Rank 8 2025 used brand and size features]
C2. LLM field extraction when regex coverage is low: small instruct model, JSON-constrained output,
    vLLM batch over train+test once, cache to parquet (brand, category, unit size, pack, premium/organic,
    material...). [Seen: Rank 80, 2025 extracted 15 fields with vLLM]
C3. Derived: total quantity = size x pack, log total quantity, alternate target log(price / total qty)
    as a diverse ensemble member, text stats (chars, words, digits, bullets), keyword flags. [General; text
    stats Seen 2023]
C4. High-cardinality ids: learned embedding inside NNs [Seen: product_type_id embedding, Rank 2, 2023];
    K-fold target encoding for GBDTs (gbdt_cv.py te=). Never encode on the full train. [General]
C5. Character n-grams for typos (gbdt_cv.py tfidf= uses word + char_wb). [Seen: char-level, 2023]
C6. Scientist basics: drop pure ids (phone, serial), drop exact-redundant columns (charges = minutes x
    rate), map yes/no → 1/0, one-hot low-cardinality categoricals. (eda.py flags all of these.)

## D. Representations

- Fine-tune end to end when GPU allows; keep frozen-embedding GBDT as a diverse member. [Seen: Rank 8, 2025]
- Numeric-aware text encoder: BERT missed numeric cues → EmbeddingGemma. [Seen: Rank 20, 2025]
- SigLIP over CLIP for stable regression embeddings (CLIP scale drift). [Seen: Rank 20, 2025; Rank 476 also SigLIP2]
- Higher image resolution helps when images carry printed text. [Seen: Rank 8, 2025]
- Second text encoder for long descriptions (CLIP text caps at 77 tokens). [Seen: Rank 8, 2025]
- Multilingual encoder for mixed-language catalogs. [Seen: Rank 16, 2021]
- Aux losses: image-text InfoNCE (0.2, tau 0.07) + SimCSE masked-view InfoNCE (0.1). [Seen: Rank 8, 2025]
- kNN / retrieval features over the training set, computed out of fold (scripts/knn_feats.py).
  [Seen: FAISS at Rank 80, retrieval regressor at Top 40, 2025; GPU kNN vote, 2021]
  Retrieval over EXTERNAL price data was banned in 2025.

## E. Fusion (increasing cost)

1. Late fusion of separate models. 2. Concat normalised embeddings + tabular → MLP [Rank 8].
3. Gated fusion g = sigmoid(W[t;i]) [2025 Medium]. 4. Dual-path cross-attention + small FFN 512-256-128
with BatchNorm + GELU [Rank 20; deeper heads overfit]. 5. Cross-modal transformer + retrieval [Top 40].
Always: zero vector + has_image flag for missing images [Rank 8]; inject parsed numbers into the head;
modality dropout [General]. Template: scripts/mm_finetune.py.

## F. VLM / LLM

| Strategy | Evidence |
|---|---|
| Strict zero-shot prompt with exact output format | Qwen2-VL-7B AWQ 0.617 [Rank 1, 2024] |
| Dynamic few-shot, 3 exemplars from a bucketed error pool | best single 70.9 [Rank 6, 2024] |
| QLoRA SFT via LLaMA-Factory (8-bit, lr 5e-5, bs 8, grad acc 8) | [Ranks 1 and 6, 2024] |
| Custom prompt + rotation augmentation; resize > 980 px | [Rank 45, 2024] |
| OCR text (PaddleOCR) appended to a small VLM's prompt | 2024 entry of the 2025 Rank 20 author |
| LLM as regressor (Qwen2.5 / Granite via Unsloth) | [Rank 80, 2025] diverse but costly |
| LLM as feature extractor | [Rank 80, 2025] best cost/benefit |
| AWQ / 4-bit / 8-bit quantised inference | fits 7B to 8B on T4/L4 |

## G. Validation

- One fold file for the whole team (make_folds.py); never change it. [General]
- Group folds by duplicate cluster (normalised title) when listings repeat. [General]
- Trust OOF over the public LB (public LB was only 25K of 75K in 2025). [Seen: Rank 80, 2025]
- Adversarial validation (adv_val.py) when OOF and LB disagree. [General]
- Log every experiment and submission with OOF and LB. [Seen: Rank 8 kept an 11-submission log]

## H. Ensembling

- Diversity beats count: vary inputs, targets, model families, losses. [General]
- Average best 2 [Rank 8, 2025]; vote across prompting strategies [Rank 6, 2024]; metric-aware min
  [Rank 2, 2023]; GBDT trio + MLP [Rank 476, 2025]; kNN vote [2021].
- Weights fitted on OOF in log space (blend.py). Check OOF error correlation; drop redundant members.

## I. Post-processing

- Enforce exact output grammar; run the organiser's sanity checker. [Ranks 6, 45, 2024]
- Fractions → decimals, ' and " → feet/inches, ranges → upper value, invalid unit → empty. [Ranks 6, 45, 2024]
- Snap to training grid; clip to range; OOF-tuned multiplier. [Rank 2, 2023; General]
- Post-processing alone gave about +3 F1. [Rank 6, 2024]

## J. Compute

- Rank 357 (2024) finished only ~35% of test inference on free quotas: plan GPUs on day 0.
- Rank 45 sharded inference over tmux sessions: 14 h → 4 h. Rank 64 spread inference over 15+ Kaggle GPUs.
- Rank 80 used vLLM batching on an A100. Download images in hour 1 (download_images.py).
- Reserve the final 12 hours for inference, blending, post-processing and the write-up.
