"""
predict.py
----------
Stage 3: Full Test Inference & Output Generation.

Production Predict Pipeline:
  - Country-partitioned execution (France -> US -> India) for minimal memory overhead.
  - Multi-Key Inverted Index blocking for ultra-fast candidate generation (< 10 candidates/S1).
  - High-throughput C++ rapidfuzz feature extraction.
  - Random Forest inference with high-precision threshold optimization.
  - Direct streaming output generation to ensure memory stability.
  - Produces:
      output/matching_results.tsv  (the file scored on leaderboard)
      output/candidate_pairs.tsv   (the candidate set submitted in final zip)
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import gc

sys.path.insert(0, os.path.dirname(__file__))
from blocking import build_inverted_index, query_candidates_for_s1
from features import compute_pair_features


def predict(model_path="models/model.pkl", output_dir="output", top_k=8, threshold=None):
    print("=" * 70)
    print("  AMAZON ML CHALLENGE 2026 — FULL PREDICTION PIPELINE")
    print("=" * 70)

    # 1. Load Model
    print(f"\n[1/4] Loading model configuration from {model_path}...")
    with open(model_path, "rb") as f:
        config = pickle.load(f)

    model = config['model']
    feature_cols = config['feature_cols']
    pred_threshold = threshold if threshold is not None else config.get('threshold', 0.90)

    print(f"  Model Type:        {type(model).__name__}")
    print(f"  Decision Threshold:{pred_threshold:.2f}")
    print(f"  Candidate Top-K:   {top_k}")
    print(f"  Validation F0.5:   {config.get('val_macro_f05', config.get('val_f05', 'N/A'))}")

    os.makedirs(output_dir, exist_ok=True)
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")

    # Initialize output TSVs with exact required headers
    with open(matching_path, "w", encoding="utf-8") as f_m:
        f_m.write("source1_entity_id\tmatched_entity_ids\n")

    with open(candidate_path, "w", encoding="utf-8") as f_c:
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")

    # 2. Load Test S1
    print("\n[2/4] Loading test_source1.tsv...")
    ts1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t")
    print(f"  Total Test S1 Entities: {len(ts1):,}")
    country_counts = ts1['country'].value_counts().to_dict()
    print(f"  Country Distribution: {country_counts}")

    # 3. Country-by-Country Processing
    # Order: France (smallest) -> US -> India
    processing_order = ['France', 'US', 'India']

    total_s1_processed = 0
    total_matches_predicted = 0
    total_candidates_generated = 0
    total_singletons_predicted = 0

    t_start_all = time.time()

    for country in processing_order:
        print(f"\n{'='*60}")
        print(f"  PROCESSING COUNTRY: {country.upper()}")
        print(f"{'='*60}")

        # S1 for this country
        s1_country = ts1[ts1['country'] == country].copy().reset_index(drop=True)
        n_s1_c = len(s1_country)
        print(f"  [S1] {n_s1_c:,} entities in {country}")
        s1_lookup = s1_country.set_index('entity_id').to_dict('index')

        # Load S2 and S3 filtered by this country
        print(f"  Loading S2 and S3 records for {country}...")
        t_load = time.time()
        s2_chunks = []
        for chunk in pd.read_csv("dataset/test/test_source2.tsv", sep="\t", chunksize=500000):
            filtered = chunk[chunk['country'] == country]
            if len(filtered) > 0:
                s2_chunks.append(filtered)
        s2_country = pd.concat(s2_chunks, ignore_index=True) if s2_chunks else pd.DataFrame()
        del s2_chunks
        gc.collect()

        s3_chunks = []
        for chunk in pd.read_csv("dataset/test/test_source3.tsv", sep="\t", chunksize=500000):
            filtered = chunk[chunk['country'] == country]
            if len(filtered) > 0:
                s3_chunks.append(filtered)
        s3_country = pd.concat(s3_chunks, ignore_index=True) if s3_chunks else pd.DataFrame()
        del s3_chunks
        gc.collect()

        cands_country = pd.concat([s2_country, s3_country], ignore_index=True)
        del s2_country, s3_country
        gc.collect()

        n_cands_c = len(cands_country)
        print(f"  [S2+S3] Loaded {n_cands_c:,} candidate records in {time.time()-t_load:.2f}s")

        cand_lookup = cands_country.set_index('entity_id').to_dict('index')

        # Build Inverted Index for this country
        print(f"  Building Inverted Index for {country}...")
        t_idx = time.time()
        index = build_inverted_index(cands_country, max_key_size=500)
        del cands_country
        gc.collect()

        # Generate candidates and predict in batches of S1 entities
        BATCH_SIZE = 50000
        n_batches = (n_s1_c + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Generating candidates & classifying in {n_batches} batches of {BATCH_SIZE:,}...")

        matching_lines = []
        candidate_lines = []

        for b in range(n_batches):
            b_start = b * BATCH_SIZE
            b_end = min(b_start + BATCH_SIZE, n_s1_c)
            s1_batch = s1_country.iloc[b_start:b_end]

            # Inverted index candidate generation
            cands_batch = query_candidates_for_s1(s1_batch, index, top_k=top_k, min_votes=1)

            # Feature computation and batch model scoring
            pairs_to_score = []
            pairs_meta = []  # (s1_id, cid)

            for s1_id, c_list in cands_batch.items():
                s1_row = s1_lookup.get(s1_id)
                if not s1_row:
                    continue
                for cid in c_list:
                    cand_row = cand_lookup.get(cid)
                    if cand_row:
                        feats = compute_pair_features(s1_row, cand_row)
                        pairs_to_score.append([feats[c] for c in feature_cols])
                        pairs_meta.append((s1_id, cid))

            # Model prediction
            matches_by_s1 = defaultdict(list)
            if pairs_to_score:
                probs = model.predict_proba(pairs_to_score)[:, 1]
                for (s1_id, cid), prob in zip(pairs_meta, probs):
                    if prob >= pred_threshold:
                        matches_by_s1[s1_id].append(cid)

            # Format TSV lines
            for rec in s1_batch.itertuples(index=False):
                s1_id = rec.entity_id
                c_list = cands_batch.get(s1_id, [])
                m_list = matches_by_s1.get(s1_id, [])

                # Format matching TSV line
                m_str = ",".join(m_list)
                matching_lines.append(f"{s1_id}\t{m_str}\n")

                # Format candidate TSV line
                c_str = ",".join(c_list)
                candidate_lines.append(f"{s1_id}\t{c_str}\n")

                total_s1_processed += 1
                total_candidates_generated += len(c_list)
                total_matches_predicted += len(m_list)
                if len(m_list) == 0:
                    total_singletons_predicted += 1

            # Stream flush to disk periodically
            if len(matching_lines) >= 25000:
                with open(matching_path, "a", encoding="utf-8") as f_m:
                    f_m.writelines(matching_lines)
                with open(candidate_path, "a", encoding="utf-8") as f_c:
                    f_c.writelines(candidate_lines)
                matching_lines.clear()
                candidate_lines.clear()

            print(f"    Batch {b+1}/{n_batches} done ({b_end:,}/{n_s1_c:,} entities)")

        # Flush remaining lines for this country
        if matching_lines:
            with open(matching_path, "a", encoding="utf-8") as f_m:
                f_m.writelines(matching_lines)
            with open(candidate_path, "a", encoding="utf-8") as f_c:
                f_c.writelines(candidate_lines)
            matching_lines.clear()
            candidate_lines.clear()

        # Clean memory for next country
        del s1_country, s1_lookup, cand_lookup, index
        gc.collect()
        print(f"  {country} completed successfully!")

    # 4. Final Summary & Validation
    t_total = time.time() - t_start_all
    print("\n" + "=" * 70)
    print("  PREDICTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"  Total S1 Entities Processed:    {total_s1_processed:,}")
    print(f"  Total Candidates Generated:     {total_candidates_generated:,}")
    print(f"  Average Candidates per S1:      {total_candidates_generated/max(total_s1_processed,1):.2f}")
    print(f"  Total Predicted Matches:        {total_matches_predicted:,}")
    print(f"  Predicted Singletons:           {total_singletons_predicted:,} ({total_singletons_predicted/max(total_s1_processed,1)*100:.1f}%)")
    print(f"  Total Elapsed Time:             {t_total/60:.2f} minutes")
    print(f"  Matching file:  {matching_path} ({os.path.getsize(matching_path):,} bytes)")
    print(f"  Candidate file: {candidate_path} ({os.path.getsize(candidate_path):,} bytes)")
    print("=" * 70)


if __name__ == '__main__':
    predict()
