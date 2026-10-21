"""Multicollinearity among the retained predictors, and refits of the procedure without depression diagnosis and
without PHQ-8 on the same training and test split."""
import os
import numpy as np, pandas as pd
from scipy.stats import pointbiserialr
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from analysis.pipeline import fit_fold, holdout_split, scale_train_test, shap_values, SEED, CONTINUOUS

OUT = "results/sensitivity"
HOLDOUT = "results/holdout"
MODELS = ["lightgbm", "lr"]
REMOVE = ["dx_depression", "phq8_total"]


def main():
    os.makedirs(OUT, exist_ok=True)
    X, y, X_tr, X_te, y_tr, y_te = holdout_split()

    sel = pd.read_csv(f"{HOLDOUT}/selected_features.csv")["feature"].tolist()
    Xs, _, _ = scale_train_test(X_tr, X_te)
    Z = Xs[sel].values
    vif = pd.DataFrame({"feature": sel, "VIF": [variance_inflation_factor(Z, i) for i in range(Z.shape[1])]})
    vif.loc[len(vif)] = ["pointbiserial_phq8_x_dx_depression", pointbiserialr(X["dx_depression"], X["phq8_total"])[0]]
    vif.to_csv(f"{OUT}/multicollinearity.csv", index=False)
    print(vif.round(3).to_string(index=False))

    rows = []
    for drop in REMOVE:
        print(f"\n=== without {drop} ===")
        r = fit_fold(X_tr.drop(columns=drop), y_tr, X_te.drop(columns=drop), algorithms=MODELS)
        s = r["selected"]
        Xall = X.drop(columns=drop).copy(); cont = [c for c in CONTINUOUS if c in Xall.columns]
        Xall[cont] = r["scaler"].transform(Xall[cont])
        full = clone(r["models"][MODELS[0]]).fit(Xall[s], y)
        imp = pd.Series(np.abs(shap_values(full, Xall[s])).mean(0), index=s).sort_values(ascending=False)
        rows.append(dict(dropped=drop, n_selected=len(s), selected=";".join(s),
                         auc_lightgbm=roc_auc_score(y_te, r["p_te"][MODELS[0]]), auc_lr=roc_auc_score(y_te, r["p_te"]["lr"]),
                         shap_top5=";".join(imp.index[:5]),
                         helplessness_rank=int(imp.rank(ascending=False)["cpss_helplessness"])))
    pd.DataFrame(rows).to_csv(f"{OUT}/sensitivity.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
