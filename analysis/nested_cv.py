"""Repeated stratified nested cross-validation on the full sample: the whole procedure is refitted within each outer
training fold and the outer test fold is scored once.

Outputs (results/nested/):
  fold_metrics.csv          AUC and Brier score per repeat, fold and algorithm
  summary_auc.csv           mean, SD and range of the fold AUCs, and pooled out-of-fold AUC and Brier score
  fold_selection.csv        predictors retained in each outer fold
  selection_frequency.csv   proportion of outer folds in which each candidate was retained
"""
import os
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from analysis.pipeline import fit_fold, load_data, SEED, ALGORITHMS

OUT = "results/nested"
N_OUTER, N_REPEATS = 5, 5


def main():
    os.makedirs(OUT, exist_ok=True)
    X, y, _ = load_data()
    cands = list(X.columns)
    rkf = RepeatedStratifiedKFold(n_splits=N_OUTER, n_repeats=N_REPEATS, random_state=SEED)
    fm, sel_rows, oof = [], [], []
    for k, (tr, te) in enumerate(rkf.split(X, y)):
        rep, fold = divmod(k, N_OUTER)
        print(f"\n=== repeat {rep} fold {fold} (train {len(tr)}, test {len(te)}) ===")
        r = fit_fold(X.iloc[tr], y.iloc[tr], X.iloc[te], seed=SEED + k)
        sel = r["selected"]
        sel_rows.append({**{c: int(c in sel) for c in cands}, "repeat": rep, "fold": fold})
        yt = y.iloc[te].values
        for a in ALGORITHMS:
            p = r["p_te"][a]
            fm.append(dict(repeat=rep, fold=fold, algorithm=a, auc=roc_auc_score(yt, p), brier=brier_score_loss(yt, p),
                           n_test=len(te), n_events=int(yt.sum()), cv_auc=r["cv_auc"][a]))
            oof.append(pd.DataFrame({"row": X.index[te], "repeat": rep, "algorithm": a, "y": yt, "p": p}))

    fm = pd.DataFrame(fm); oofd = pd.concat(oof)
    fm.to_csv(f"{OUT}/fold_metrics.csv", index=False)
    pd.DataFrame(sel_rows).to_csv(f"{OUT}/fold_selection.csv", index=False)
    summ = fm.groupby("algorithm")["auc"].agg(["mean", "std", "min", "max"])
    pooled = oofd.groupby(["algorithm", "row"]).agg(y=("y", "first"), p=("p", "mean")).reset_index()
    summ["pooled_oof_auc"] = pooled.groupby("algorithm").apply(lambda g: roc_auc_score(g.y, g.p))
    summ["pooled_oof_brier"] = pooled.groupby("algorithm").apply(lambda g: brier_score_loss(g.y, g.p))
    summ.sort_values("mean", ascending=False).to_csv(f"{OUT}/summary_auc.csv")
    print("\nNested cross-validation:\n" + summ.round(3).to_string())

    sf = pd.DataFrame(sel_rows)[cands].mean().rename("selection_frequency")
    sf.sort_values(ascending=False).to_csv(f"{OUT}/selection_frequency.csv")


if __name__ == "__main__":
    main()
