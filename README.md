
# IPSA V3 — Leakage-Aware Experimental Evaluation

## Purpose
V1 used a benchmark whose class labels were perfectly separated by password length. A length-only decision tree also achieved 100%, demonstrating label leakage. V2 used frequency-rank exposure tiers but engineered features alone underperformed the length baseline.

V3 evaluates four models on the same held-out data:
1. Length-only baseline.
2. IPSA engineered features **excluding length**.
3. Character-level TF-IDF (2–5 character n-grams).
4. Combined character TF-IDF + IPSA engineered features.

The target is **empirical guessability/exposure tier**, not an absolute password-security score:
- 0: top 10% most frequent passwords (high exposure)
- 1: next 40% (medium exposure)
- 2: bottom 50% of the downloaded corpus (lower exposure)

## Run
Open CMD in this folder:
```bat
python -m pip install -r requirements_v3.txt
python download_rockyou.py
python train_v3.py
```

Outputs:
- `model_v3/metrics.json`
- `model_v3/classification_reports.txt`
- `model_v3/confusion_matrix_combined.csv`
- `model_v3/ipsa_v3_combined.joblib`

## Research caution
This is a benchmark evaluation, not evidence of real-world cracking resistance. RockYou-derived data is breached-password data and should be used only for authorized academic/security research. Do not publish plaintext password records. Report aggregate metrics only.

Do not claim that V3 is "more accurate" unless its test metrics actually exceed the stated baseline. Do not use any metric before the script produces it on the held-out test set.
