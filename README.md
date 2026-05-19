# 🏀 NBA Game Outcome Prediction

### Machine Learning Pipeline for Pre-Game Win Probability Estimation

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Best%20Model-brightgreen?style=flat-square)
![AUC](https://img.shields.io/badge/Best%20AUC-~0.700-blue?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-14%2C118%20game%20records-orange?style=flat-square)
![Seasons](https://img.shields.io/badge/NBA%20Seasons-2019%E2%80%932024-red?style=flat-square)

---

## The Problem That Made This Project Harder

The first model scored **98.1% accuracy**. It was wrong.

Post-game statistics — Offensive Rating, Defensive Rating, Points Scored — had leaked into the feature set. Since `ORtg > DRtg` is mathematically equivalent to `Win = 1`, the model was learning the scoreline identity, not team form. It would have been useless on any real game.

Identifying, diagnosing, and correcting that leakage is the core contribution of this project. The corrected pipeline delivers honest predictions at **~65–66% accuracy / ~0.700 AUC** — consistent with published sports analytics benchmarks of 63–70% for team-level NBA data.

---

## Project Summary

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

**🏟️ Home Court is the Dominant Signal**
`Is_Home` carries 2–3× the predictive weight of any rolling statistic. Home teams win ~58% of NBA games — a 16 percentage-point edge. It is the single most important feature across all seven models.

**🎯 Shooting Efficiency Predicts Winning**
Rolling True Shooting % (`TS%`) and Effective FG% (`eFG%`) are the strongest rolling predictors. Teams in shooting slumps are measurably more likely to lose, independent of opponent quality.

**🔄 Turnovers Are the Strongest Negative Signal**
Rolling turnover rate (`TOV%`) is the top negative predictor. Poor ball security gives opponents extra possessions and correlates directly with reduced win probability.

**😴 Back-to-Back Fatigue is Real and Measurable**
Teams on zero days' rest win only **44%** of games — 6 percentage points below average. This validates load management strategies increasingly adopted across the league.

**📊 Gradient Boosting Dominates Tabular Data**
LightGBM and CatBoost outperform all other classifiers. Their ability to model non-linear interactions (e.g. high `TS%` at home vs. away) gives them a consistent edge over linear models.

---

## Dataset

**Source:** [Basketball-Reference.com](https://www.basketball-reference.com) — NBA Team Game Logs (2019–2024)

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

> Temporal split mirrors real deployment — train on past seasons, predict future games the model has never seen. This prevents any future-data contamination.

---

## Repository Structure

```
Basketball-Game-Outcome-Prediction/
│
├── Dataset/
│   └── 2019_2024_nba.csv
│
├── Notebook/
│   ├── NEW_NBA_Prediction_v2.ipynb       # Final corrected pipeline
│   ├── nba_prediction_code.ipynb         # Original (leaky) model — kept for reference
│   └── nba_scrapping.ipynb               # Data collection from Basketball-Reference
│
├── Report/
│   ├── Hrushikesh_Mate_DA_Thesis.pdf
│   └── New_NBA_Project_Report.pdf
│
└── README.md
```

---

## Methodology

### 1. Data Cleaning & Leakage Removal

| Step | Action | Reason |
|---|---|---|
| 1 | Parse dates; sort by Season, Team, Date | Correct rolling window order |
| 2 | Drop `OPts`, `DPts` | Post-game scores — direct leakage |
| 3 | Drop `ORtg`, `DRtg` | Post-game ratings — primary leakage source |
| 4 | Encode `Location` → `Is_Home` (1/0) | Numeric for all models |
| 5 | Compute `Days_Rest` per team | Game fatigue signal |
| 6 | Drop duplicate columns (`Rk`, `G`, `At`) | Confirmed redundant |

### 2. Feature Engineering

Since every raw statistic is measured **after** the game, each column is shifted into a rolling mean over the previous N games — using `.shift(1)` so the current game's own result is excluded from its own prediction vector.

- **12 base statistics** × **2 window sizes** (5-game, 10-game) = **24 rolling features**
- **5 context features**: `Is_Home`, `Days_Rest`, `Team`, `Opp_Team`, `Season`
- **Total**: 29 features

Missing values in the first 5–10 rows per team are imputed using `SimpleImputer(strategy='median')` fitted exclusively on the training set.

### 3. Leakage-Safe Pipeline

All preprocessing is wrapped in a `sklearn.Pipeline`:

```
OrdinalEncoder (Team, Opp_Team) → SimpleImputer → StandardScaler → Classifier
```

Encoders and scalers are fitted only on training folds during cross-validation. No test-set statistics influence preprocessing.

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

> **Primary metric: ROC-AUC** — more meaningful than accuracy on a balanced dataset since it measures discrimination across all decision thresholds.

### Visual Outputs

**ROC Curve — All Models**
![ROC Curve](./roc_curve.png)

**Confusion Matrix — LightGBM**
![Confusion Matrix](./confusion_matrix.png)

---

## SHAP Explainability

SHAP (SHapley Additive exPlanations) applied to LightGBM. Values are additive — they sum to the model's output log-odds.

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

> No single feature exceeds ~0.08 SHAP importance, reflecting the genuine complexity of NBA outcomes. The model combines many weak signals — which is why ensemble methods outperform linear classifiers.

**SHAP Summary Plot**
![SHAP Summary](./shap_summary.png)

---

## Limitations & Future Work

| Limitation | Impact |
|---|---|
| No opponent rolling features — only own-team stats used | Medium — opponent form not captured beyond identity encoding |
| No player-level data — injuries, lineup composition absent | High — a star player absence can shift win probability by 10–15pp |
| Not benchmarked against Vegas implied probabilities | Medium — no comparison to sharpest market consensus |
| Fixed rolling windows (5 and 10 games) | Low-Medium — optimal window may vary by statistic |

**What would actually improve this:**
- Add opponent rolling stats as mirrored features — the model currently ignores who you're playing against
- Integrate injury reports and lineup data via NBA API to capture the biggest single game-level signal the model currently misses
- Benchmark predicted probabilities against closing Vegas lines to measure edge vs. market consensus

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
| Environment | Python 3.12 / Jupyter Notebook |
| Data Source | [Basketball-Reference.com](https://www.basketball-reference.com) |

---

## Contact

**Hrushikesh Mate**
📍 Dublin, Ireland
🔗 [LinkedIn](https://www.linkedin.com/in/your-linkedin-here)
🐙 [GitHub](https://github.com/HrushhikeshMate)
