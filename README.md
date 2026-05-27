# startup-investment-prediction-cross-market
### A Comparative ML Study of Shark Tank US, Shark Tank India & Dragons' Den UK

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange) ![License](https://img.shields.io/badge/License-MIT-green) ![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## Overview

This project is the **first three-way cross-market ML comparison** of entrepreneurial TV investment shows. Four classifiers were trained on three culturally distinct markets to predict funding outcomes and identify what separates funded from unfunded startups — universally and per market.

| Market | Samples | Deal Rate | Best Model | F1 | AUC |
|---|---|---|---|---|---|
| Shark Tank US | 1,481 | 61.7% | Logistic Regression | 0.747 | 0.600 |
| Shark Tank India | 789 | 63.4% | XGBoost | **0.926** | **0.944** |
| Dragons' Den UK | 1,473 | 35.3% | XGBoost | 0.796 | 0.925 |

---

## Research Questions

- **RQ1 (Primary):** What features consistently predict investment outcomes across US, India, and UK — and which are market-specific?
- **RQ2:** How does model performance vary across markets under the same experimental framework?
- **RQ3:** Does SMOTE improve prediction performance consistently across markets?

---

## Key Findings

1. **Universal predictors** — Ask Amount, Equity Offered (%), and Valuation Requested appear in the top-5 permutation importance features across all three markets.
2. **India is the most predictable** (AUC = 0.944) due to rich pre-pitch financial KPIs (EBITDA, gross margin, monthly revenue). The US is the hardest (AUC = 0.600).
3. **Cross-market transfer fails** — all 9 transfer configurations (train on 2, test on 1) show negative F1 delta, confirming investment decisions are partly culturally embedded.
4. **SMOTE is unreliable** — hurts Logistic Regression on the US by −0.14 F1. Threshold tuning on the precision-recall curve is more consistent and risk-free.
5. **Operational impact** — India's XGBoost correctly identifies 55+ true deals per 100 pitches vs ~38 for random guessing.

---

## Models Used

| Model | Key Hyperparameters |
|---|---|
| Logistic Regression | `max_iter=1000, C=1.0` |
| Random Forest | `n_estimators=200, class_weight=balanced` |
| SVM | `kernel=rbf, probability=True, class_weight=balanced` |
| XGBoost | `n_estimators=200, learning_rate=0.05` |

All models evaluated with **stratified 5-fold CV**, **paired t-tests**, **calibration analysis**, and **optimal threshold tuning via precision-recall curve**.

---

## Project Structure

```
cross-market-investment-prediction/
│
├── notebooks/
│   └── ML_INVESTMENT_PREDICTION.ipynb   ← main notebook
├── data/
│   └── README.md                                  ← dataset download instructions
├── requirements.txt
└── README.md
```

---

## Setup & Run

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/cross-market-investment-prediction.git
cd cross-market-investment-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download datasets**

See [`data/README.md`](data/README.md) for Kaggle download links.
Update the `FILE_PATHS` dictionary in Cell 4 of the notebook to point to your local files.

**4. Run the notebook**
```bash
jupyter notebook notebooks/ML_INVESTMENT_PREDICTION.ipynb
```

Run all cells top to bottom. All figures are saved automatically to the working directory.

---

## Tech Stack

- **Language:** Python 3.10
- **ML:** scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Data:** pandas, numpy
- **Visualisation:** matplotlib, seaborn
- **Stats:** scipy (paired t-test, Mann-Whitney U)
- **Environment:** Jupyter Notebook
---

## Results Highlights

**Cross-market transfer experiment** (train on 2 markets, test on 3rd):

| Train | Test | RF ΔF1 | XGB ΔF1 | LR ΔF1 |
|---|---|---|---|---|
| US + India | UK | −0.20 | −0.24 | −0.16 |
| US + UK | India | −0.35 | −0.31 | −0.19 |
| India + UK | US | −0.08 | −0.10 | −0.06 |

All deltas are negative — investment decisions do not transfer freely across markets.

---

## Author
Smit Savaliya