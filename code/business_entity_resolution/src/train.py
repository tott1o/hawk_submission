"""
train.py
--------
Stage 2B: Train the ML Classifier on Real Candidate Pairs.

Production Training Pipeline:
  1. Loads ground truth and real S1 records from train_source1.tsv.
  2. Builds candidate inverted index from train_source2.tsv and train_source3.tsv.
  3. Generates realistic candidate pairs (positives + blocking hard negatives).
  4. Computes 15 fuzzy similarity features (rapidfuzz).
  5. Trains Random Forest classifier with balanced class weighting.
  6. Optimizes threshold directly for Macro F_0.5 score on held-out validation S1 entities.
  7. Saves model configuration to models/model.pkl.
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import gc

sys.path.insert(0, os.path.dirname(__file__))
from blocking import build_inverted_index, query_candidates_for_s1
from features import compute_pair_features


def f05_score_single(true_set, pred_set):
    """Compute Macro F_0.5 for a single S1 entity."""
    if len(true_set) == 0 and len(pred_set) == 0:
        return 1.0
    if len(true_set) == 0 or len(pred_set) == 0:
        return 0.0
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    if prec + rec == 0:
        return 0.0
    return (1.25 * prec * rec) / (0.25 * prec + rec)


def train(n_train_s1=100000, n_val_s1=20000, n_cand_rows=1500000, output_model_path="models/model.pkl"):
    print("=" * 70)
    print("  AMAZON ML CHALLENGE 2026 — PRODUCTION TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load Ground Truth
    print(f"\n[1/6] Loading Ground Truth ({n_train_s1 + n_val_s1:,} entities)...")
    gt_all = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=n_train_s1 + n_val_s1)
    gt_train = gt_all.iloc[:n_train_s1].copy()
    gt_val   = gt_all.iloc[n_train_s1:].copy()

    gt_train_dict = {}
    for _, row in gt_train.iterrows():
        s1_id = row['source1_entity_id']
        m_str = str(row['matched_entity_ids'])
        gt_train_dict[s1_id] = set(m_str.split(',')) if m_str and m_str != 'nan' else set()

    gt_val_dict = {}
    for _, row in gt_val.iterrows():
        s1_id = row['source1_entity_id']
        m_str = str(row['matched_entity_ids'])
        gt_val_dict[s1_id] = set(m_str.split(',')) if m_str and m_str != 'nan' else set()

    # 2. Load S1 Records
    needed_s1_ids = set(gt_all['source1_entity_id'])
    print(f"\n[2/6] Loading {len(needed_s1_ids):,} S1 records from train_source1.tsv...")
    s1_records = {}
    for chunk in pd.read_csv('dataset/train/train_source1.tsv', sep='\t', chunksize=250000):
        found = chunk[chunk['entity_id'].isin(needed_s1_ids)]
        for _, r in found.iterrows():
            s1_records[r['entity_id']] = r.to_dict()
        if len(s1_records) >= len(needed_s1_ids):
            break
    print(f"  Loaded {len(s1_records):,} S1 records.")

    # 3. Load Candidate Pool (S2 and S3)
    print(f"\n[3/6] Loading candidate pool ({n_cand_rows:,} rows each from S2 & S3)...")
    s2_df = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', nrows=n_cand_rows)
    s3_df = pd.read_csv('dataset/train/train_source3.tsv', sep='\t', nrows=n_cand_rows)
    cands_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    gc.collect()

    cand_lookup = cands_df.set_index('entity_id').to_dict('index')
    print(f"  Total candidate pool: {len(cand_lookup):,} records.")

    # 4. Build Inverted Index
    print("\n[4/6] Building Inverted Index for candidate generation...")
    index = build_inverted_index(cands_df, max_key_size=500)
    del cands_df
    gc.collect()

    # 5. Build Training Feature Dataset
    print("\n[5/6] Generating candidate pairs and extracting features...")
    # Sample 30k S1 entities for training pairs
    train_sample_ids = list(gt_train['source1_entity_id'])[:30000]
    s1_train_sub = pd.DataFrame([s1_records[i] for i in train_sample_ids if i in s1_records])

    train_cands = query_candidates_for_s1(s1_train_sub, index, top_k=8)

    train_rows = []
    train_labels = []

    for s1_id, c_list in tqdm(train_cands.items(), desc="  Features"):
        s1_row = s1_records.get(s1_id)
        if not s1_row:
            continue
        true_set = gt_train_dict.get(s1_id, set())

        # Include positive matches in training set
        candidates_to_score = list(c_list)
        for t_id in true_set:
            if t_id in cand_lookup and t_id not in candidates_to_score:
                candidates_to_score.append(t_id)

        for cid in candidates_to_score:
            cand_row = cand_lookup.get(cid)
            if cand_row:
                feats = compute_pair_features(s1_row, cand_row)
                train_rows.append(feats)
                train_labels.append(1 if cid in true_set else 0)

    train_df = pd.DataFrame(train_rows)
    train_df['label'] = train_labels
    feature_cols = [c for c in train_df.columns if c != 'label']

    pos = int(train_df['label'].sum())
    neg = int((train_df['label'] == 0).sum())
    print(f"  Training pairs: {len(train_df):,} | Pos: {pos:,} ({pos/len(train_df)*100:.1f}%) | Neg: {neg:,} ({neg/len(train_df)*100:.1f}%)")

    # Train Random Forest
    print("\n[6/6] Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(train_df[feature_cols], train_df['label'])

    # Feature importances
    feat_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nFeature Importances:")
    for fname, imp in feat_imp.items():
        print(f"  {fname:<20}: {imp:.4f}")

    # Threshold Optimization on Validation Set
    print("\nOptimizing decision threshold on 5,000 validation S1 entities...")
    val_sample_ids = list(gt_val['source1_entity_id'])[:5000]
    s1_val_sub = pd.DataFrame([s1_records[i] for i in val_sample_ids if i in s1_records])

    val_cands = query_candidates_for_s1(s1_val_sub, index, top_k=8)

    val_features = []
    val_meta = []

    for s1_id, c_list in val_cands.items():
        s1_row = s1_records.get(s1_id)
        if not s1_row:
            continue
        for cid in c_list:
            cand_row = cand_lookup.get(cid)
            if cand_row:
                feats = compute_pair_features(s1_row, cand_row)
                val_features.append([feats[c] for c in feature_cols])
                val_meta.append((s1_id, cid))

    val_probs = model.predict_proba(val_features)[:, 1]

    best_thresh = 0.85
    best_f05 = 0.0
    print("\nThreshold  |  Macro F0.5  |  Precision  |  Recall")
    print("-----------+--------------+-------------+--------")

    for t in np.arange(0.70, 0.96, 0.05):
        pred_dict = {s1_id: [] for s1_id in val_sample_ids}
        for (s1_id, cid), p in zip(val_meta, val_probs):
            if p >= t:
                pred_dict[s1_id].append(cid)

        scores = [f05_score_single(gt_val_dict.get(s1_id, set()) & set(cand_lookup.keys()),
                                   set(pred_dict.get(s1_id, [])))
                  for s1_id in val_sample_ids]
        m_score = np.mean(scores)

        tp = sum(len(gt_val_dict.get(s1_id, set()) & set(cand_lookup.keys()) & set(pred_dict[s1_id])) for s1_id in val_sample_ids)
        fp = sum(len(set(pred_dict[s1_id]) - (gt_val_dict.get(s1_id, set()) & set(cand_lookup.keys()))) for s1_id in val_sample_ids)
        fn = sum(len((gt_val_dict.get(s1_id, set()) & set(cand_lookup.keys())) - set(pred_dict[s1_id])) for s1_id in val_sample_ids)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0

        marker = " <-- BEST" if m_score > best_f05 else ""
        if m_score > best_f05:
            best_f05 = m_score
            best_thresh = t
        print(f"{t:.2f}       |  {m_score:.4f}      |  {prec:.4f}     |  {rec:.4f}{marker}")

    print(f"\nFinal Chosen Threshold: {best_thresh:.2f} (Macro F0.5 = {best_f05:.4f})")

    # Save to model path
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    config = {
        'model': model,
        'feature_cols': feature_cols,
        'threshold': best_thresh,
        'val_macro_f05': best_f05,
        'top_k': 8,
    }
    with open(output_model_path, 'wb') as f:
        pickle.dump(config, f)
    print(f"Model saved to {output_model_path}!")
    return config


if __name__ == '__main__':
    train()
