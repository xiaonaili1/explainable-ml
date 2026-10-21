"""Held-out evaluation of the seven algorithms, SHAP interpretation, rank agreement and the adjusted logistic model."""
import os, json
import numpy as np, pandas as pd, shap
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, brier_score_loss, balanced_accuracy_score, f1_score, confusion_matrix
from statsmodels.api import Logit, add_constant
from statsmodels.stats.multitest import multipletests
from analysis.pipeline import (fit_fold, holdout_split, shap_values, SEED, CONTINUOUS, ALGORITHMS, TREE_MODELS)

OUT = "results/holdout"
N_BOOT = 1000


def bootstrap_ci(y, p, fn, B=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed); v = []
    for _ in range(B):
        i = rng.choice(len(y), len(y), replace=True)
        v.append(fn(y[i], p[i]))
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def metrics_row(a, y, p):
    yp = (p >= 0.5).astype(int); tn, fp, fn_, tp = confusion_matrix(y, yp).ravel()
    lo, hi = bootstrap_ci(y, p, roc_auc_score)
    return dict(algorithm=a, auc=roc_auc_score(y, p), auc_lo=lo, auc_hi=hi, brier=brier_score_loss(y, p),
                bacc=balanced_accuracy_score(y, yp), sens=tp / (tp + fn_), spec=tn / (tn + fp),
                ppv=tp / (tp + fp), npv=tn / (tn + fn_), f1=f1_score(y, yp))


def kendall_w(rank_matrix):
    n, k = rank_matrix.shape
    S = ((rank_matrix.sum(1) - rank_matrix.sum() / n) ** 2).sum()
    return 12 * S / (k ** 2 * (n ** 3 - n))


def main():
    os.makedirs(OUT, exist_ok=True)
    X, y, X_tr, X_te, y_tr, y_te = holdout_split()
    print(f"{len(X)} participants, {X.shape[1]} candidates; training {len(X_tr)} ({y_tr.sum()} events), test {len(X_te)} ({y_te.sum()} events)")

    res = fit_fold(X_tr, y_tr, X_te)
    sel = res["selected"]
    pd.Series(sel, name="feature").to_csv(f"{OUT}/selected_features.csv", index=False)
    json.dump({a: {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()}
               for a, p in res["best_params"].items()}, open(f"{OUT}/best_params.json", "w"), indent=1)

    # test-set performance
    yv = y_te.values
    preds = pd.DataFrame({"y_true": yv}, index=X_te.index)
    rows = []
    for a in ALGORITHMS:
        preds[f"prob_{a}"] = res["p_te"][a]; rows.append(metrics_row(a, yv, res["p_te"][a]))
    met = pd.DataFrame(rows).sort_values("auc", ascending=False)
    met.to_csv(f"{OUT}/metrics.csv", index=False); preds.to_parquet(f"{OUT}/test_predictions.parquet")
    print("\nTest-set performance:\n" + met.round(3).to_string(index=False))

    # interpretation model: the tree-based model with the highest test-set AUC, refitted on the full sample
    tree_met = met[met.algorithm.isin(TREE_MODELS)].sort_values(["auc", "brier"], ascending=[False, True])
    top = tree_met.iloc[0]["algorithm"]; print(f"\nInterpretation model: {top}")
    X_s = X.copy(); cont = [c for c in CONTINUOUS if c in X.columns]
    X_s[cont] = res["scaler"].transform(X[cont])
    full = clone(res["models"][top]).fit(X_s[sel], y)
    sv = shap_values(full, X_s[sel])
    np.savez(f"{OUT}/shap_full_refit.npz", shap_values=sv, features=np.array(sel), model=np.array(top))
    imp_full = pd.Series(np.abs(sv).mean(0), index=sel).sort_values(ascending=False)
    imp_test = pd.Series(np.abs(shap_values(res["models"][top], res["X_te_s"])).mean(0), index=sel)

    # agreement of the SHAP rankings across the seven algorithms
    Xtr_s = res["X_tr_s"]; ranks = {}
    bg = shap.kmeans(Xtr_s, 50)
    for a in ALGORITHMS:
        m = res["models"][a]
        if a in TREE_MODELS:
            v = shap_values(m, Xtr_s)
        elif a == "lr":
            v = shap.LinearExplainer(m, Xtr_s).shap_values(Xtr_s)
        else:
            f = lambda Z, m=m: m.predict_proba(pd.DataFrame(Z, columns=sel))[:, 1]
            v = shap.KernelExplainer(f, bg).shap_values(Xtr_s.sample(200, random_state=SEED), nsamples=200, silent=True)
        ranks[a] = pd.Series(np.abs(np.asarray(v)).mean(0), index=sel).rank(ascending=False)
    R = pd.DataFrame(ranks)
    W = kendall_w(R.values)
    R["kendall_W"] = W; R.to_csv(f"{OUT}/rank_agreement.csv")
    print(f"\nKendall's W = {W:.3f}")

    # adjusted logistic regression on the full sample, continuous predictors in SD units
    Z = X[sel].copy(); cz = [c for c in cont if c in sel]; Z[cz] = (Z[cz] - Z[cz].mean()) / Z[cz].std(ddof=0)
    r = Logit(y, add_constant(Z)).fit(disp=0, maxiter=5000)
    t2 = pd.DataFrame({"feature": sel, "OR": np.exp(r.params[1:].values),
                       "OR_lo": np.exp(r.conf_int().iloc[1:, 0].values), "OR_hi": np.exp(r.conf_int().iloc[1:, 1].values),
                       "p": r.pvalues[1:].values})
    t2["fdr_p"] = multipletests(t2["p"], method="fdr_bh")[1]
    t2 = t2.set_index("feature").join(imp_full.rename("mean_abs_shap_full_refit")).join(imp_test.rename("mean_abs_shap_holdout_test"))
    t2["shap_rank"] = t2["mean_abs_shap_full_refit"].rank(ascending=False).astype(int)
    t2.sort_values("shap_rank").to_csv(f"{OUT}/table2.csv")
    print("\nTable 2:\n" + t2.sort_values("shap_rank").round(3).to_string())


if __name__ == "__main__":
    main()
