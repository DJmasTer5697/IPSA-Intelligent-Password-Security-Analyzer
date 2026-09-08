# IPSA Prototype

**Intelligent Password Security Analyzer (IPSA)**

This prototype implements the methodology described in the RIC paper:
- password feature extraction
- entropy estimation
- dictionary/sequential/repetition/keyboard pattern indicators
- Random Forest classification
- Accuracy, Precision, Recall, F1-score
- confusion matrix
- feature importance
- Streamlit user interface

## Dataset

Use a CSV with:
```text
password,strength
example123,0
Example@2026,1
...
```

Recommended public benchmark: the Password Strength Classifier dataset (about 669k labeled passwords) is described in multiple public ML studies. Its labels are pre-existing strength categories, so results should be described as classification performance on that benchmark, not as proof of real-world cracking resistance.

## Run

```bash
pip install -r requirements.txt
python train_model.py --data passwords.csv
streamlit run app.py
```

Training produces:
- `model/metrics.json`
- `model/classification_report.txt`
- `model/confusion_matrix.png`
- `model/feature_importance.csv`
- `model/ipsa_random_forest.joblib`

## Research caution

Do not put accuracy/precision/recall/F1 values into the paper until the actual training script has been run on the final dataset. Also, because many public password-strength datasets have labels derived from existing strength rules/meters, the paper should avoid claiming that the classifier proves resistance to real password cracking unless an independent attack-resistance evaluation is performed.
