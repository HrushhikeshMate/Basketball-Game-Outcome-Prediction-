# 🏀 NBA Game Outcome Prediction
### A Machine Learning Approach to Pre-Game Win Probability Estimation

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Best%20Model-brightgreen?style=flat-square)
![AUC](https://img.shields.io/badge/Best%20AUC-~0.700-blue?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-14%2C118%20game%20records-orange?style=flat-square)
![Seasons](https://img.shields.io/badge/NBA%20Seasons-2019–2024-red?style=flat-square)
![Status](https://img.shields.io/badge/Status-Academic%20Submission-purple?style=flat-square)

> **MSc Data Analytics — Dublin Business School | Academic Year 2024–2025**  
> Author: Hrushikesh Mate (Roll No. 20025400)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [SHAP Explainability](#shap-explainability)
- [Limitations & Future Work](#limitations--future-work)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Overview

This project builds an end-to-end machine learning pipeline to predict **NBA game outcomes before tip-off** — using only pre-game rolling statistics. It covers the full data science lifecycle: raw data ingestion, leakage-free feature engineering, model training and evaluation, SHAP explainability, probability calibration analysis, and live single-game inference.

A critical contribution of this project is the **identification and correction of data leakage** present in a prior model iteration. Post-game statistics (Offensive Rating, Defensive Rating, Points) were inadvertently used as predictive features, inflating model accuracy to an unrealistic **98.1%**. The corrected pipeline delivers honest, deployable predictions at **~65–66% accuracy / ~0.69–0.70 AUC** — consistent with published sports analytics benchmarks of 63–70% AUC for team-level NBA data.

| Metric | Value |
|---|---|
| Dataset Size | 14,118 game-team rows |
| Seasons Covered | 2019–2024 (6 NBA seasons) |
| Features Used | 29 (rolling + contextual) |
| Models Trained | 7 (including Stacking Ensemble) |
| Best Single Model | LightGBM (~65% accuracy, ~0.693 AUC) |
| Best Overall | Hybrid Stacking Ensemble (~66% accuracy, ~0.700 AUC) |

---

## Key Findings

**🚨 Data Leakage Eliminated**  
The original model scored 98.1% accuracy by inadvertently using post-game statistics (`ORtg`, `DRtg`, `OPts`, `DPts`) as input features. Since `ORtg > DRtg` is mathematically equivalent to `Win = 1`, the model was learning the score identity — not team form. Corrected accuracy: **63–66%**.

**🏟️ Home Court is the Dominant Signal**  
`Is_Home` carries 2–3× the predictive weight of any rolling statistic. Home teams win ~58% of NBA games — a 16 percentage-point edge over away teams. This is the single most important feature across all seven models.

**🎯 Shooting Efficiency Predicts Winning**  
Rolling True Shooting % (`TS%`) and Effective FG% (`eFG%`) are the strongest rolling predictors. Teams in shooting slumps are measurably more likely to lose, independent of opponent quality.

**🔄 Turnovers Are the Strongest Negative Signal**  
Rolling turnover rate (`TOV%`) is the most important negative predictor. Poor ball security gives opponents extra possessions and correlates significantly with reduced win probability.

**😴 Back-to-Back Fatigue is Real and Measurable**  
Teams on zero days' rest win only **44%** of games — 6 percentage points below average. This validates NBA teams' increasing use of load management for star players on second-night back-to-backs.

**📊 Gradient Boosting Dominates Tabular Data**  
LightGBM and CatBoost outperform all other classifiers. Their ability to model non-linear interactions (e.g. high `TS%` at home vs. high `TS%` away) gives them a systematic edge over linear models.

---

## Dataset

**Source:** [Basketball-Reference.com](https://www.basketball-reference.com) — NBA Team Game Logs (2019–2024)  
**File:** `2019_2024_nba.csv` (compiled from per-game team box score pages)

| Property | Detail |
|---|---|
| Coverage | 6 complete NBA seasons: 2018-19 through 2023-24 |
| Rows | 14,118 (each game appears twice — once per team) |
| Original Columns | 32 |
| Final Features | 29 (after dropping leaky and redundant columns) |
| Target | `Win`: binary (1 = Win, 0 = Loss) |
| Class Balance | Perfectly balanced at 50% win rate across all seasons |

### Train / Test Split (Temporal)

| Split | Seasons | Rows | Share |
|---|---|---|---|
| Training | 2019, 2020, 2021, 2022 | 10,280 | ~73% |
| Test | 2023, 2024 | 3,838 | ~27% |

> **Rationale:** A temporal split mirrors real deployment — training on past seasons and predicting future games the model has never seen. This prevents any future-data contamination.

---

## Project Structure

```
nba-game-prediction/
│
├── Dataset/
│   └── 2019_2024_nba.csv
│
├── Notebook/
│   ├── NEW_NBA_Prediction_v2.ipynb
│   ├── nba_prediction_code.ipynb
│   └── nba_scrapping.ipynb
│
├── Report/
│   ├── Hrushikesh_Mate_DA_Thesis.pdf
│   └── New_NBA_Project_Report.pdf
│
└── README.md
```

---

## Installation & Setup

### Prerequisites

- Python 3.12+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/HrushhikeshMate/Basketball-Game-Outcome-Prediction-.git

cd Basketball-Game-Outcome-Prediction
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
**Core dependencies:**

```
pandas
numpy
scikit-learn
lightgbm
catboost
shap
plotly
matplotlib
seaborn
jupyter
```

### 4. Place the Dataset

Download the NBA game log CSV from [Basketball-Reference.com](https://www.basketball-reference.com) and place it at:

```
data/2019_2024_nba.csv
```

---

## Usage

### Run the Full Notebook

```bash
jupyter notebook notebooks/NEW_NBA_Prediction.ipynb
```

The notebook walks through every stage: data cleaning → feature engineering → EDA → model training → evaluation → SHAP analysis → calibration.

### Single-Game Win Probability Prediction

Use the `predict_game()` function to estimate win probability for any upcoming game using only pre-game rolling statistics:

```python
from src.predict import predict_game

result = predict_game(
    team="GSW",
    opponent="LAL",
    is_home=1,
    days_rest=2,
    ts_pct_roll5=0.592,
    efg_pct_roll5=0.568,
    tov_pct_roll5=12.3,
    trb_pct_roll5=51.2,
    # ... (all 12 rolling stats for 5-game and 10-game windows)
)

print(f"Predicted Win Probability: {result['win_prob']:.1%}")
print(f"Predicted Outcome: {result['prediction']}")
# → Predicted Win Probability: 56.0%
# → Predicted Outcome: WIN
```

> The function uses the trained LightGBM model (best single model) and accepts rolling averages computed from the team's most recent games. No post-game data is required or used.


## Methodology

### 1. Data Cleaning

| Step | Action | Reason |
|---|---|---|
| 1 | Parse dates; sort by Season, Team, Date | Correct rolling window order |
| 2 | Drop `OPts`, `DPts` | Post-game scores — direct leakage |
| 3 | Drop `ORtg`, `DRtg` | Post-game ratings — primary leakage source |
| 4 | Encode `Location` → `Is_Home` (1/0) | Numeric for all models |
| 5 | Compute `Days_Rest` per team | Game fatigue signal |
| 6 | Drop duplicate columns (`Rk`, `G`, `At`) | Confirmed redundant by assertion |

### 2. Feature Engineering

Since every raw statistic is measured **after** the game, the solution is to shift each column into a rolling mean over the previous N games:

- **12 base statistics** × **2 window sizes** (5-game, 10-game) = **24 rolling features**
- **5 context features**: `Is_Home`, `Days_Rest`, `Team`, `Opp_Team`, `Season`
- **Total**: 29 features

Rolling features use `.shift(1)` so the current game's own result is excluded from its own prediction vector.

Missing values in the first 5–10 rows per team (insufficient rolling history) are imputed using `SimpleImputer(strategy='median')` fitted exclusively on the training set.

### 3. Leakage-Safe Pipeline

All preprocessing is wrapped in a `sklearn.Pipeline`:

```
OrdinalEncoder (Team, Opp_Team) → SimpleImputer → StandardScaler → Classifier
```

Encoders and scalers are **fitted only on training folds** during cross-validation. No test-set statistics ever influence preprocessing.

---

## Model Performance

All models evaluated on the temporal test set (Seasons 2023–2024):

| Model | Test Accuracy | Test ROC-AUC | Test F1 | CV AUC (mean ± std) |
|---|---|---|---|---|
| 🥇 Hybrid Stacking Ensemble | ~66% | ~0.700 | ~0.66 | 0.700 ± 0.010 |
| 🥈 LightGBM | ~65% | ~0.693 | ~0.65 | 0.693 ± 0.009 |
| 🥉 CatBoost | ~64% | ~0.685 | ~0.64 | 0.685 ± 0.010 |
| Random Forest | ~63% | ~0.672 | ~0.63 | 0.672 ± 0.011 |
| AdaBoost | ~61% | ~0.641 | ~0.61 | 0.641 ± 0.012 |
| Logistic Regression | ~60% | ~0.630 | ~0.60 | 0.630 ± 0.010 |
| Decision Tree | ~58% | ~0.592 | ~0.58 | 0.592 ± 0.018 |

**Primary metric: ROC-AUC** (measures discrimination across all thresholds — more meaningful than accuracy on a balanced dataset).

### Probability Calibration

| Model | Calibration Quality | Notes |
|---|---|---|
| Logistic Regression | ✅ Excellent | Best natural calibration — recommended for broadcast overlays |
| CatBoost | ✅ Very Good | Minor isotonic adjustment if needed |
| LightGBM | ✅ Good | Slight overconfidence at extremes — apply isotonic regression for deployment |
| Random Forest | ⚠️ Moderate | Platt scaling recommended |
| Decision Tree | ❌ Poor | Not suitable for probability applications |

---

## SHAP Explainability

SHAP (SHapley Additive exPlanations) was applied to LightGBM to explain predictions at the feature level. Values are additive — they sum to the model's output log-odds.

| Feature | Mean SHAP | Direction | Interpretation |
|---|---|---|---|
| `Is_Home` | 0.082 | ↑ Positive | Home court — crowd, familiarity, reduced travel |
| `TS_pct_roll5` | 0.061 | ↑ Positive | Recent shooting efficiency → offensive edge |
| `eFG_pct_roll5` | 0.054 | ↑ Positive | Three-point shooting efficiency |
| `TOV_pct_roll10` | 0.047 | ↓ Negative | Turnovers gift possessions to opponents |
| `Team` (ordinal) | 0.038 | Variable | Franchise quality baseline |
| `DRB_pct_roll5` | 0.031 | ↑ Positive | Defensive rebounding limits second chances |
| `Days_Rest` | 0.025 | ↓ at 0 | Back-to-back fatigue penalty |
| `ORB_pct_roll5` | 0.020 | ↑ Slight | Offensive rebounding — second-chance scoring |

> No single feature has SHAP importance above ~0.08, reflecting the genuine complexity of NBA game outcomes. The model combines many weak signals — which is why ensemble methods outperform linear classifiers.


---

## Limitations & Future Work

### Current Limitations

| Limitation | Impact |
|---|---|
| No opponent rolling features — only own-team stats used | Medium — opponent form not captured beyond identity encoding |
| No player-level data — injuries, load management, lineup composition absent | High — a single star player absence can shift win prob by 10–15pp |
| Fixed rolling windows (5 and 10 games) — optimal window may vary by statistic | Low-Medium |
| Not benchmarked against Vegas implied probabilities | Medium — no comparison to sharpest market consensus |


---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Manipulation | `pandas`, `NumPy` |
| Visualisation | `Plotly Express`, `Matplotlib`, `Seaborn` |
| ML Preprocessing | `scikit-learn` (Pipeline, OrdinalEncoder, StandardScaler, SimpleImputer) |
| Modelling | `LightGBM`, `CatBoost`, `scikit-learn` (LR, DT, RF, AdaBoost) |
| Ensemble | `sklearn.ensemble.StackingClassifier` |
| Explainability | `SHAP` (TreeExplainer) |
| Calibration | `sklearn.calibration.calibration_curve` |
| Environment | Python 3.12 / Jupyter Notebook / VS Code |
| Data Source | [Basketball-Reference.com](https://www.basketball-reference.com) |

---

## Academic Context

This project was submitted as part of the **MSc Data Analytics** programme at **Dublin Business School**, Academic Year 2024–2025.

It demonstrates competencies in:
- End-to-end ML pipeline design
- Data leakage identification and prevention
- Temporal validation methodology
- Ensemble modelling and hyperparameter tuning
- Model interpretability with SHAP
- Probability calibration analysis
- Production-ready inference function design

---

## License

This project is submitted for academic purposes. The dataset is sourced from [Basketball-Reference.com](https://www.basketball-reference.com) and is subject to their terms of use. Code in this repository is available for educational reference.

---

## Contact

**Hrushikesh Mate**  
MSc Data Analytics — Dublin Business School  
Roll No. 20025400  
📍 Dublin, Ireland

---

*"A model that predicts 70% win probability should correspond to teams actually winning 70% of those games."*  
— Probability calibration is not optional.
