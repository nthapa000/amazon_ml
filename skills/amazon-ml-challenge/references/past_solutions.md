# Past editions: what ranked teams did (self-reported)

## Problem history
| Year | Task | Data | Metric |
|---|---|---|---|
| 2021 | browse node classification | ~2.9M train, 9,919 classes | accuracy |
| 2023 | product length regression | ~2.25M train, ~735K test | max(0, 100(1 − MAPE)) |
| 2024 | entity value extraction from images | ~264K train, ~131K test | exact-match F1 |
| 2025 | smart product pricing (text + image) | 75K train, 75K test | SMAPE |
No 2022 edition was found. Trend: text → image → multimodal; always catalog data, always a non-MSE metric.

## 2025 (SMAPE; winner ≈ 39.7)
- Rank 8 CTRL+ALT+DEV (40.78): CLIP ViT-L/14-336 + DistilBERT, fine-tuned end to end; Huber on log2 price +
  0.2 CLIP InfoNCE + 0.1 SimCSE; lr 2e-5, 15 epochs, warmup 6%, wd 0.01, clip 1.0, FP16; missing images
  zero-padded; average of best two. Ladder: Qwen3-emb+LGBM 57.40, LGBM+features 51.49, e5 50.70,
  DistilBERT FT 45.43, CLIP 42.23, CLIP-336 41.54, final 40.78.
  https://github.com/VishalTheHuman/Amazon-ML-Challenge-2025/
- Rank 20: EmbeddingGemma (BERT missed numbers), SigLIP 2 Giant (CLIP scale drift), dual-path
  cross-attention, 512-256-128 FFN, direct SMAPE loss, AdamW cosine.
  https://medium.com/nybles/amazon-applied-scientist-intern-interview-experience-ml-challenge-2025-5092441d0b2e
- Top 40 TensorTitans: cross-modal transformer + retrieval regressor + Optuna. https://github.com/raptor7197/amazon-ML-school
- Rank 80 ML Mavericks (43.28): 30+ notebooks; vLLM 15-field LLM extraction; FAISS; Qwen2.5/Granite via
  Unsloth; "validation-test gap fixes" ensemble; lesson: data understanding + trusted local validation.
  https://github.com/NeelDevenShah/Amazon-ML-Challenge-2025
- Rank 476: SigLIP2 + MiniLM + engineered feats → LGBM/XGB/CatBoost + MLP.
  https://github.com/Spkap/amazon-ml-challenge-2025-Multimodal-Product-Price-Prediction

## 2024 (exact-match F1)
- Rank 1 NeuralNinjas: Qwen2-VL-7B; zero-shot 0.617 → QLoRA 10K 0.678 → 20K 0.679 → 1.6K curated 0.865.
  LLaMA-Factory, 8-bit QLoRA, bs 8, lr 5e-5, grad acc 8. https://github.com/KhadgaA/Amazon-ML-Challenge
- Rank 6 DBkaScam: ZSP 66.2 → +post-proc 69.3; SFT 150K 64.8; dynamic few-shot 70.9; vote 71.8.
  https://github.com/arnav10goel/Amazon-ML-Challenge-24
- Rank 45: Idefics2-8B; 0.44 → 0.56 → 0.616 with custom prompt + rotation; tmux sharding 14 h → 4 h.
  https://github.com/hwaseem04/Amazon-ML-Challenge-2024
- Rank 64: MiniCPM-Llama3-V-2.5, inference over 15+ Kaggle GPUs. https://github.com/Sneh-T-Shah/Amazon-ML-Challenge-2024
- Rank 357: Moondream + regex on free quotas; only ~35% of test processed.

## 2023 (MAPE score)
- Rank 2 Team Fishes: log target, clip at 12, z-score; concatenated text; BERT/RoBERTa FT with learned
  product_type_id embedding; snap to nearest train value; minimum of two models.
  https://github.com/greenfish8090/AmazonML
- Rank 118: GBM, 35.13 (shows the transformer + metric-trick gap).

## 2021 (accuracy)
- Winners: no_bert_train_challenge (IIT ISM), DeVaSh.Ai (IIT Guwahati), Patanjali Noodles; key: use the
  entire training set. Rank 16: mBERT, full data, 3 epochs, 67.47. kNN + vote 66.85. LinearSVC on 50K: 55.48.

# Learning resources

| Topic | Link |
|---|---|
| Session code (churn notebook) | https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_applying_machine_learning/xgboost_customer_churn/xgboost_customer_churn.ipynb |
| SageMaker XGBoost input format | https://github.com/awsdocs/amazon-sagemaker-developer-guide/blob/master/doc_source/xgboost.md |
| SageMaker tuning | https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html |
| SageMaker Autopilot | https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-automate-model-development.html |
| XGBoost missing values | https://xgboost.readthedocs.io/en/stable/faq.html |
| StatQuest XGBoost Part 1 / 2 | https://www.youtube.com/watch?v=OtD8wVaFm6E , https://www.youtube.com/watch?v=8b1JEDvenQU |
| StatQuest ROC and AUC | https://www.youtube.com/watch?v=4jRBRDbJemM |
| Threshold tuning | https://scikit-learn.org/stable/modules/classification_threshold.html |
| Feature engineering / target encoding | https://www.kaggle.com/learn/feature-engineering |
| Adversarial validation | https://www.kaggle.com/code/lukeimurfather/adversarial-validation-train-vs-test-distribution |
| Ensembling guide | https://mlwave.com/kaggle-ensembling-guide/ |
| Optuna | https://github.com/optuna/optuna |
| LoRA / QLoRA explained | https://www.youtube.com/watch?v=t1caDsMzWBk |
| Unsloth notebooks (incl. Qwen2-VL) | https://github.com/unslothai/notebooks |
| LLaMA-Factory | https://github.com/hiyouga/LLaMA-Factory |
