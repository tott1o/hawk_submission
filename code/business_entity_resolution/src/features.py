"""
features.py
-----------
Stage 2A: Feature Engineering.

For every (Source 1 entity, candidate) pair produced by blocking,
compute a set of numeric similarity features. These features are
then fed into the ML classifier (Random Forest / XGBoost).
"""

import pandas as pd
import numpy as np
from rapidfuzz import fuzz

from normalize import normalize_name, normalize_address


def compute_pair_features(row1, row2):
    """
    Compute string-similarity features between two business records.

    Args:
        row1: dict or Series with 'business_name', 'business_address', 'country'
        row2: dict or Series with 'business_name', 'business_address', 'country'

    Returns:
        dict of feature_name -> float value (all values in [0, 1])
    """
    # Normalize text
    name1 = normalize_name(row1.get('business_name', ''))
    name2 = normalize_name(row2.get('business_name', ''))
    addr1 = normalize_address(row1.get('business_address', ''))
    addr2 = normalize_address(row2.get('business_address', ''))

    features = {}

    # ── NAME FEATURES ────────────────────────────────────────────────────────
    # ratio: simple character-level edit distance ratio
    #   "mcdonalds" vs "mcdonald" → ~0.94
    features['name_ratio']      = fuzz.ratio(name1, name2) / 100.0

    # partial_ratio: best match of the shorter string inside the longer string
    #   "kfc" vs "kfc fast food" → 1.0
    features['name_partial']    = fuzz.partial_ratio(name1, name2) / 100.0

    # token_sort_ratio: sorts words alphabetically before comparing
    #   "blue star hotel" vs "hotel blue star" → 1.0
    features['name_token_sort'] = fuzz.token_sort_ratio(name1, name2) / 100.0

    # token_set_ratio: handles one name being a subset of another
    #   "amazon" vs "amazon india private limited" → 1.0
    features['name_token_set']  = fuzz.token_set_ratio(name1, name2) / 100.0

    # Best of all name scores (take the maximum)
    features['name_best']       = max(
        features['name_ratio'],
        features['name_partial'],
        features['name_token_sort'],
        features['name_token_set']
    )

    # ── ADDRESS FEATURES ─────────────────────────────────────────────────────
    features['addr_ratio']      = fuzz.ratio(addr1, addr2) / 100.0
    features['addr_partial']    = fuzz.partial_ratio(addr1, addr2) / 100.0
    features['addr_token_sort'] = fuzz.token_sort_ratio(addr1, addr2) / 100.0
    features['addr_token_set']  = fuzz.token_set_ratio(addr1, addr2) / 100.0

    features['addr_best']       = max(
        features['addr_ratio'],
        features['addr_partial'],
        features['addr_token_sort'],
        features['addr_token_set']
    )

    # ── COMBINED FEATURES ────────────────────────────────────────────────────
    # Simple average of best name score and best address score
    features['combined_avg']    = (features['name_best'] + features['addr_best']) / 2.0

    # Product of best name and best address (both must be high)
    features['combined_product'] = features['name_best'] * features['addr_best']

    # ── COUNTRY MATCH ────────────────────────────────────────────────────────
    # 1.0 if same country, 0.0 if different (strong signal!)
    c1 = str(row1.get('country', '')).strip().lower()
    c2 = str(row2.get('country', '')).strip().lower()
    features['country_match']   = 1.0 if (c1 == c2 and c1 != '') else 0.0

    # ── LENGTH RATIO FEATURES ────────────────────────────────────────────────
    # Ratio of shorter length to longer length (0 = very different lengths)
    len_n1, len_n2 = len(name1), len(name2)
    len_a1, len_a2 = len(addr1), len(addr2)
    features['name_len_ratio']  = min(len_n1, len_n2) / (max(len_n1, len_n2) + 1)
    features['addr_len_ratio']  = min(len_a1, len_a2) / (max(len_a1, len_a2) + 1)

    return features


def build_feature_dataset(s1, s2, s3, candidates, gt=None):
    """
    Build a DataFrame of features for all candidate pairs.
    If gt (ground truth) is provided, adds a 'label' column (1=match, 0=no match).
    Used for training. For test data, call without gt.

    Args:
        s1, s2, s3:   Source DataFrames
        candidates:   dict from blocking.run_blocking()
        gt:           ground truth DataFrame (optional, for training only)

    Returns:
        features_df:  DataFrame with feature columns + optional 'label'
        pair_ids:     list of (s1_entity_id, candidate_entity_id) tuples
    """
    from tqdm import tqdm

    # Build fast lookup dicts: entity_id -> row dict
    s1_lookup = s1.set_index('entity_id').to_dict('index')
    s2_lookup = s2.set_index('entity_id').to_dict('index')
    s3_lookup = s3.set_index('entity_id').to_dict('index')

    # Build ground truth lookup if provided
    gt_dict = None
    if gt is not None:
        gt_dict = {}
        for _, row in gt.iterrows():
            s1_id = row['source1_entity_id']
            matches_str = row.get('matched_entity_ids', '')
            if pd.notna(matches_str) and str(matches_str).strip():
                gt_dict[s1_id] = set(str(matches_str).split(','))
            else:
                gt_dict[s1_id] = set()

    rows     = []
    pair_ids = []

    print("\n[FEATURES] Computing features for all candidate pairs...")
    for s1_id, cand_list in tqdm(candidates.items(), desc="  Features"):
        s1_row = s1_lookup.get(s1_id)
        if s1_row is None:
            continue

        true_matches = gt_dict.get(s1_id, set()) if gt_dict else set()

        for cand_id in cand_list:
            # Look up the candidate in the correct source
            if cand_id.startswith('S2'):
                cand_row = s2_lookup.get(cand_id)
            else:
                cand_row = s3_lookup.get(cand_id)

            if cand_row is None:
                continue

            feats = compute_pair_features(s1_row, cand_row)

            if gt_dict is not None:
                feats['label'] = 1 if cand_id in true_matches else 0

            rows.append(feats)
            pair_ids.append((s1_id, cand_id))

    features_df = pd.DataFrame(rows)
    print(f"  Total pairs: {len(features_df):,}")
    if 'label' in features_df.columns:
        pos = features_df['label'].sum()
        neg = (features_df['label'] == 0).sum()
        print(f"  Positive (match=1): {pos:,} ({pos/len(features_df)*100:.1f}%)")
        print(f"  Negative (match=0): {neg:,} ({neg/len(features_df)*100:.1f}%)")

    return features_df, pair_ids


if __name__ == '__main__':
    # Quick test of the features
    r1 = {'business_name': "McDonald's Corp.", 'business_address': "1 Main St, New York", 'country': 'US'}
    r2 = {'business_name': "McDonalds Corporation", 'business_address': "Main Street 1, NYC", 'country': 'US'}
    r3 = {'business_name': "Starbucks", 'business_address': "5th Avenue, New York", 'country': 'US'}

    print("Match pair (should be high scores):")
    feats = compute_pair_features(r1, r2)
    for k, v in feats.items():
        print(f"  {k}: {v:.3f}")

    print("\nNon-match pair (should be low scores):")
    feats2 = compute_pair_features(r1, r3)
    for k, v in feats2.items():
        print(f"  {k}: {v:.3f}")
