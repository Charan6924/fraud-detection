# Missing Value Strategy — IEEE-CIS Fraud Detection

## Context

The IEEE-CIS dataset has ~400 columns with block-wise missing patterns — entire groups of V features are missing together, the identity table is only available for 24% of transactions, and most id_* columns are 90%+ null. A uniform imputation strategy doesn't make sense here. The plan is to tier columns by missing rate and handle each group differently.

## Strategy

### Tier 1 — Drop (>80% missing)
These columns are too sparse to carry signal and dropping them reduces dimensionality noise.

- V138–V166 (86% missing)
- V322–V339 (86% missing)
- D6–D14 (83–93% missing, except D10 at 12.9%)
- dist2 (93.6%)
- R_emaildomain (76.8%) — borderline but 77% is too sparse to encode usefully
- id_03, id_04, id_07, id_08, id_09, id_10, id_18, id_21–id_27, id_30, id_32, id_33, id_34 (all >44% missing within the identity subset, and only 24% of transactions have identity data at all, making effective missing rate >85%)

### Tier 2 — Keep as NaN (XGBoost handles natively)
XGBoost learns the optimal split direction for missing values at each node, so no imputation needed.

- V1–V11 (47.3% missing)
- V12–V34 (12.9% missing)
- V35–V52 (28.6% missing)
- V53–V94 (13–15% missing)
- V167–V278 (76.4% missing) — borderline but keep, XGBoost can use them when present
- D1–D5 (0.2–52% missing)
- D10, D15 (12.9–15.1% missing)
- dist1 (59.7% missing)
- id_01, id_02, id_05, id_06, id_11–id_17, id_19, id_20, id_28, id_29, id_31, id_35–id_38
- M1–M9 (28–59% missing)

### Tier 3 — Impute categoricals (<20% missing)
- ProductCD — mode ("W")
- card4, card6 — mode ("visa", "debit")
- P_emaildomain — fill "Unknown"
- addr1, addr2 — fill with -1

### Tier 4 — Impute low-missing numerics (<5% missing)
- card2, card3, card5 — fill -1 (they're categorical codes, not continuous)
- V95–V137 (0.1% missing)
- V279–V321 (<0.1% missing)

### Special: Identity table
- Add binary `has_identity` column (1 if identity join succeeded, 0 otherwise)
- This alone is a strong signal (mobile fraud 10.2% vs desktop 6.5%)

### Special: TransactionAmt
- Log transform (log1p) to reduce right skew

## Implementation Plan

### Files to create/modify

1. **ml/features/build_features.py** — rewrite with the full feature pipeline
   - Load both CSVs
   - Merge on TransactionID (left join)
   - Apply column drop list (Tier 1)
   - Add has_identity flag
   - Impute categoricals (Tier 3) and low-missing numerics (Tier 4)
   - Log-transform TransactionAmt
   - Standardize numeric features
   - Save as data/features.parquet

2. **ml/training/train.py** — same as before, loads features.parquet, trains XGBoost

### Verification
1. Run python ml/features/build_features.py — confirm no errors
2. Check output shape has expected columns (original - dropped + has_identity)
3. Run python ml/training/train.py — confirm training completes and PR-AUC is reasonable
