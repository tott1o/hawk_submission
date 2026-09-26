# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Hawk  
**Team Members:**  
1. Fathima Rasha (Team Leader, +91 8592894513)  
2. Akhilnath G S (+91 8921143616)  
3. Megha Singh (+91 9286181372)  
4. Mohammed Nihal A A (+91 8891619145)  
**Submission Date:** 26 September 2026  

---

## 1. Executive Summary

We present a scalable, high-precision two-stage machine learning system for enterprise-scale Business Entity Resolution across noisy, heterogeneous data sources without shared identifiers. Our architecture combines **Country-Partitioned Multi-Key Inverted Index Blocking** with an ensemble **Random Forest Classifier** trained on realistic candidate pairs with hard negative mining. By pruning the search space from $17.3 \times 10^{12}$ comparisons to an average of under 8 candidate pairs per Source 1 entity, our pipeline achieves an internal validation **Macro $F_{0.5}$ score of 0.8101** (with 90.9% precision), while executing complete end-to-end inference across all 1.73 million test entities in under 15 minutes with a peak RAM footprint under 2.5 GB.

---

## 2. Methodology

### 2.1 Problem Analysis
In enterprise platforms, business identity data originates from independent, uncoordinated registries, web directories, and commercial feeds. Exploratory data analysis (EDA) across the training and test splits ($2.2\text{M}$ training S1, $5.0\text{M}$ S2, $5.3\text{M}$ S3, and $1.73\text{M}$ test S1 records) revealed several prominent noise patterns:

1. **Orthographic and Transliteration Noise**:
   - In India, many company names in Source 2 and Source 3 appear in non-Latin scripts (e.g., Devanagari: `एसएस फूड प्राइवेट लिमिटेड`, Tamil: `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி`), whereas the corresponding address retains English alphanumeric tokens and street names (`AF-0684`, `6(29) C.I.T. Colony`).
   - In the test set, France introduces accented characters (`é`, `è`, `ê`, `à`, `ç`) and specific corporate designations (`SARL`, `SAS`, `SCI`, `EURL`).
2. **Structural & Word-Order Transpositions**:
   - Company names frequently exhibit inverted word order (e.g., `XX Apex Nippon` vs. `XX Nippon Apex`, `Star Trusted Table` vs. `Trusted Table Star`).
   - Corporate legal suffixes vary arbitrarily (`Corp`, `Corporation`, `Inc.`, `LLC`, `Private Limited`, `Pvt Ltd`, `LLP`).
3. **Address Inconsistencies & Partial Fields**:
   - Address strings often have missing street names, landmark-based descriptions (`Near SBI ATM`), missing PIN/ZIP codes, and hyphenated house number ranges (`1056-1060 Belden Ave` vs. `1056c Belden Ave`).
4. **Extreme Class Imbalance & Singletons**:
   - Over 50% of Source 1 entities have **no matching record** in Source 2 or Source 3 (singletons).
   - Under the macro-averaged $F_{0.5}$ metric, predicting a false merge on a true singleton drops its entity score from $1.0$ directly to $0.0$. Because $F_{0.5}$ weights precision twice as heavily as recall ($\beta = 0.5$), false positives are catastrophically penalized.

### 2.2 Solution Strategy
**Approach Type:** Scalable Inverted Index Multi-Key Blocking + RapidFuzz Feature Engineering + Cost-Sensitive Random Forest Matcher.

**Core Innovations:**
1. **Multi-Key Inverted Index with Specificity Weighting**: Rather than relying on quadratic all-to-all dense/sparse matrix comparisons ($O(N \times M)$) which fail at Amazon scale, we construct an inverted hash index mapping composite semantic keys to entity IDs. Candidates accumulate votes based on key specificity, ensuring small candidate sets ($\approx 5\text{--}8$ candidates per entity) with high recall ($>89\%$).
2. **Sorted Multi-Gram & Street Number Tokenization**: We generate order-invariant sorted 2-gram name keys and extract normalized house/plot numbers paired with distinctive street words. This captures transposed company names and multi-lingual records where address numbers remain consistent.
3. **Hard Negative Mining on Real Blocking Candidates**: Rather than training on random cross-database pairs, our model is trained directly on the candidate pairs output by the blocking stage, forcing the classifier to learn fine-grained discriminative features between same-brand entities at different locations.
4. **Direct Macro-$F_{0.5}$ Threshold Optimization**: We perform a post-training threshold sweep on validation S1 entities to locate the exact probability threshold ($\tau = 0.90$) that maximizes the competition's macro-averaged $F_{0.5}$ metric.

---

## 3. Candidate Generation (Blocking)

### 3.1 Blocking Keys Used
To ensure high recall while generating an ultra-compact candidate set, we construct 5 complementary blocking key families within each country partition:

| Key Type | Structure / Example | Primary Invariance Handled |
| :--- | :--- | :--- |
| **Sorted Name 2-Grams** | `country_n2_{tokenA}_{tokenB}` (e.g., `US_n2_apex_nippon`) | Word-order transposition (`Apex Nippon` == `Nippon Apex`) |
| **Salient Name Tokens** | `country_n1_{token}` for distinctive tokens of length $\ge 5$ | Distinctive corporate identity with omitted sub-brands |
| **Name Prefix Pairs** | `country_np_{prefA}_{prefB}` (4-char prefixes) | Minor typos, phonetic spelling errors, inflectional suffixes |
| **Address Number + Street Word** | `country_addr_{num}_{street_word}` (e.g., `US_addr_3315_fremont`) | Matching physical locations when business names diverge |
| **Postal Code Keys** | `country_pin_{code}_{name_token}` (6-digit IN / 5-digit US/FR) | Multi-lingual records (e.g., Tamil/Hindi name with English postal code) |

### 3.2 Candidate Selection & Scaling
- **Candidate pairs generated:** Average of **$5.8$ candidates per Source 1 entity** on the test set.
- **Generic Key Pruning**: Any blocking key that maps to $> 500$ entities is pruned as a generic noise key (e.g., generic words like `group`, `holdings`, `associates`).
- **Vote Accumulation**: For each Source 1 entity, candidate records accumulate votes based on key specificity: highly specific keys ($\le 50$ matches) contribute 2 votes, while broader keys contribute 1 vote. The top-8 candidates with $\ge 1$ votes are retained.
- **Scale**: The entire candidate generation stage across all 1.73 million test entities executes in under **4 minutes** and requires less than 2 GB of RAM.

---

## 4. Matching Model

### 4.1 Feature Engineering (15 Discriminative Features)
For each generated candidate pair, we extract 15 similarity features computed via C++ `rapidfuzz`:

1. **Name Ratio**: Levenshtein distance ratio normalized to $[0, 1]$.
2. **Name Partial Ratio**: Maximum fuzzy substring matching score (handles DBA names and legal suffix removals).
3. **Name Token Sort Ratio**: Token-level Levenshtein similarity after alphabetical sorting.
4. **Name Token Set Ratio**: Jaccard-style token intersection and remainder matching (handles company names where one is a strict subset of the other).
5. **Name Best**: $\max(\text{Ratio}, \text{Partial}, \text{TokenSort}, \text{TokenSet})$.
6. **Address Ratio**: Character-level edit similarity of normalized addresses.
7. **Address Partial Ratio**: Substring address matching (crucial for partial/landmark addresses).
8. **Address Token Sort Ratio**: Order-invariant address token similarity.
9. **Address Token Set Ratio**: Subset address token similarity.
10. **Address Best**: $\max(\text{AddrRatio}, \text{AddrPartial}, \text{AddrTokenSort}, \text{AddrTokenSet})$.
11. **Combined Average**: $(\text{NameBest} + \text{AddrBest}) / 2$.
12. **Combined Product**: $\text{NameBest} \times \text{AddrBest}$ (heavily penalizes pairs that match on name but completely disagree on address).
13. **Country Match**: Binary indicator ($1.0$ if country identical, $0.0$ otherwise).
14. **Name Length Ratio**: $\min(|N_1|, |N_2|) / (\max(|N_1|, |N_2|) + 1)$.
15. **Address Length Ratio**: $\min(|A_1|, |A_2|) / (\max(|A_1|, |A_2|) + 1)$.

### 4.2 Model Type & Hyperparameters
- **Architecture**: `RandomForestClassifier` with 150 estimators, max depth 12, min samples per leaf 4, and `class_weight='balanced'`.
- **Feature Importance**:
  - `combined_avg`: 23.5%
  - `combined_product`: 14.6%
  - `addr_best`: 14.4%
  - `addr_token_set`: 12.1%
  - `addr_partial`: 10.5%
  - `addr_token_sort`: 5.7%
- **Threshold Selection**: Because macro $F_{0.5}$ penalizes false positives twice as heavily as false negatives, the standard $0.5$ classification threshold produces excessive false merges on singletons. By evaluating macro $F_{0.5}$ across thresholds $\tau \in [0.50, 0.95]$ on held-out validation entities, the optimal operating threshold was determined to be **$\tau = 0.90$**, yielding 90.9% precision.

---

## 5. Results & Error Analysis

### 5.1 Validation Results
Evaluated on a held-out validation split of 5,000 Source 1 entities against the ground truth:

| Metric | Score |
| :--- | :--- |
| **Macro $F_{0.5}$ Score** | **0.8101** |
| **Pair-Level Precision** | **0.9090** |
| **Pair-Level Recall** | **0.7232** |
| **Blocking Recall Ceiling** | **89.37%** |
| **Avg. Candidates per S1 Entity** | **5.8** |
| **Predicted Singletons** | **68.4%** |

### 5.2 Error Analysis
- **Common False Positives (Wrong Merges)**:
  - Chain businesses sharing identical trade names with adjacent street addresses (e.g., two branches of a medical practice or bank on different floors of a complex).
  - Addressed by enforcing the `combined_product` feature and raising the decision threshold to $0.90$.
- **Common False Negatives (Missed Matches)**:
  - Severe transliteration mismatches where names are rendered entirely in regional scripts (e.g. Odia, Telugu) while addresses contain non-standard municipal abbreviations.
  - Mitigated by our address-based number-street blocking keys which link records independently of name language.

---

## 6. Conclusion

We developed an end-to-end, highly scalable Entity Resolution architecture tailored for the Amazon ML Challenge 2026. By eschewing computationally intractable all-to-all comparisons in favor of Multi-Key Inverted Index Blocking, our solution maintains an average candidate set of under 8 candidates per entity—directly targeting Amazon's ranking criteria that reward smaller candidate sets. Combined with rapid feature extraction and a cost-sensitive Random Forest matcher optimized for macro $F_{0.5}$, our pipeline produces robust, high-precision entity resolution across multi-million record datasets in minutes.

---

## Appendix

### A. Code Artefacts
The complete runnable pipeline is organized as follows:
```
code/business_entity_resolution/
├── src/
│   ├── normalize.py           # Unicode-aware text normalization & legal suffix expansion
│   ├── blocking.py            # Multi-Key Inverted Index candidate generator
│   ├── features.py            # 15 rapidfuzz string similarity feature extractors
│   ├── train.py               # Model training & Macro F_0.5 threshold optimizer
│   └── predict.py             # Streaming country-partitioned test inference pipeline
├── README.md                  # Step-by-step reproduction instructions
└── requirements.txt           # Pinned dependencies (pandas, scikit-learn, rapidfuzz)
```

**Reproduction Command:**
```bash
# Step 1: Train the model
python code/business_entity_resolution/src/train.py

# Step 2: Generate final predictions
python code/business_entity_resolution/src/predict.py

# Step 3: Validate outputs against official rules
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
