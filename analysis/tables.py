"""Participant characteristics by PHQ-9 item 9 response category (Table 1) and the sample values quoted in the text.

Outputs (results/tables/):
  table1_characteristics.csv   median [IQR] or n (%) of every candidate predictor, overall and by response category,
                               with Kruskal-Wallis or chi-square tests (Monte Carlo exact test when expected counts < 5)
  sample_values.json           age distribution, outcome counts, sex-stratified correlations, bivariate summary
"""
import os, json
import numpy as np, pandas as pd
from scipy import stats
from analysis.pipeline import load_data

OUT = "results/tables"
LABELS = {0: "Not at all", 1: "Several days", 2: "More than half the days", 3: "Nearly every day"}


def exact_p(tab, rng, B=20000):
    obs = stats.chi2_contingency(tab, correction=False)[0]; r = tab.sum(1); c = tab.sum(0); cnt = 0
    for _ in range(B):
        t = stats.random_table(r, c, seed=rng).rvs(1)[0]
        if stats.chi2_contingency(t, correction=False)[0] >= obs - 1e-9: cnt += 1
    return (cnt + 1) / (B + 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    X, y, d = load_data()
    si = d["outcome_si_ordinal"].astype(int)
    values = {}
    a = X["age"]
    values["age"] = {"min": float(a.min()), "max": float(a.max()), "median": float(a.median()),
                     "q1": float(a.quantile(.25)), "q3": float(a.quantile(.75)),
                     "pct_18_24": float(((a >= 18) & (a < 25)).mean() * 100),
                     "pct_25_34": float(((a >= 25) & (a < 35)).mean() * 100), "pct_35_plus": float((a >= 35).mean() * 100)}
    values["outcome"] = {"n_si": int(y.sum()), "pct_si": float(y.mean() * 100), "ordinal_counts": si.value_counts().sort_index().tolist()}
    male = (X.gender_female == 0) & (X.gender_other == 0); female = X.gender_female == 1
    values["sex_stratified_spearman_helplessness"] = {
        "male": stats.spearmanr(X.cpss_helplessness[male], si[male])[0], "female": stats.spearmanr(X.cpss_helplessness[female], si[female])[0],
        "n_male": int(male.sum()), "n_female": int(female.sum()), "n_other": int(X.gender_other.sum())}
    values["missing_values"] = int(X.isna().sum().sum())

    cont = [v for v in X.columns if X[v].nunique() > 2]
    binary = [v for v in X.columns if X[v].nunique() <= 2]
    rng = np.random.default_rng(0)
    groups = [("Overall", np.ones(len(X), bool))] + [(LABELS[k], (si == k).values) for k in range(4)]
    rows = []
    for v in cont:
        row = {"variable": v, "type": "continuous"}
        for g, mask in groups:
            x = X.loc[mask, v]; row[g] = f"{x.median():.1f} [{x.quantile(.25):.1f}-{x.quantile(.75):.1f}]"
        row["p"] = stats.kruskal(*[X.loc[(si == k).values, v] for k in range(4)])[1]; row["test"] = "Kruskal-Wallis"
        rows.append(row)
    for v in binary:
        row = {"variable": v, "type": "binary"}
        for g, mask in groups:
            x = X.loc[mask, v]; row[g] = f"{int(x.sum())} ({x.mean() * 100:.1f})"
        tab = pd.crosstab(si, X[v]).values; chi2, p, dof, exp = stats.chi2_contingency(tab)
        if exp.min() < 5: row["p"] = exact_p(tab, rng); row["test"] = "Fisher exact (Monte Carlo)"
        else: row["p"] = p; row["test"] = "Chi-square"
        rows.append(row)
    t = pd.DataFrame(rows); t.to_csv(f"{OUT}/table1_characteristics.csv", index=False)
    values["bivariate"] = {"n_candidates": len(t), "n_p_lt_05": int((t.p < 0.05).sum()), "not_significant": t.loc[t.p >= 0.05, "variable"].tolist()}
    json.dump(values, open(f"{OUT}/sample_values.json", "w"), indent=1, default=float)
    print(json.dumps(values, indent=1, default=float))
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
