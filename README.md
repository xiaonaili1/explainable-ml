# Analysis code

## Scripts

Run each command from this folder. 

| Command | What it does | Writes to |
|---|---|---|
| `python -m analysis.tables` | Table 1 and the sample values quoted in the text | `results/tables/` |
| `python -m analysis.holdout` | Stratified 80/20 split; standardisation, LASSO selection and Optuna tuning on the training set; evaluation of the seven algorithms on the test set; SHAP interpretation; rank agreement; adjusted logistic regression (Table 2) | `results/holdout/` |
| `python -m analysis.sensitivity` | Variance inflation factors; refits without depression diagnosis and without PHQ-8 | `results/sensitivity/` |
| `python -m analysis.nested_cv` | 5 x 5 repeated nested cross-validation on the full sample | `results/nested/` |
| `python -m analysis.splines` and `Rscript analysis/splines.R` | Restricted cubic spline models | `results/splines/` |
| `python -m analysis.figures` | Figures 1, 2, S1 and S2 | `results/figures/` |
| `python -m analysis.figure3` | Figure 3 | `results/figures/` |

## Results included

| Manuscript | File |
|---|---|
| Table 1 | `results/tables/table1_characteristics.csv` |
| Table 2 | `results/holdout/table2.csv` |
| Table S1 | `results/holdout/best_params.json` |
| Results 3.2, retained predictors | `results/holdout/selected_features.csv` |
| Results 3.3, test-set performance; Figure 1 | `results/holdout/metrics.csv` |
| Results 3.3, nested cross-validation | `results/nested/summary_auc.csv`, `selection_frequency.csv` |
| Results 3.4, Kendall's W | `results/holdout/rank_agreement.csv` |
| Results 3.4, correlation, VIF and refits | `results/sensitivity/` |
| Results 3.4 and Figure 3 legend, spline tests | `results/splines/` |
| Figures 1 to 3, S1 and S2 | `results/figures/` |
