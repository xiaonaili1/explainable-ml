"""Restricted cubic spline logistic models for the leading continuous predictors, equivalent to rms::lrm with rcs()
and anova() in R: five knots at the 5th, 27.5th, 50th, 72.5th and 95th percentiles, Wald tests for the overall
association and for the nonlinear terms. Each model is adjusted for the other covariates, leaving out its own target."""
import os
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score
from analysis.pipeline import DATA, OUTCOME

OUT = "results/splines"
COVARIATES = ["cpss_helplessness", "phq8_total", "gad10_total", "screen_time", "trauma_count",
              "pa_level", "gender_female", "education_years"]
TARGETS = ["phq8_total", "cpss_helplessness", "gad10_total", "cpss_self_efficacy"]
KNOT_PROBS = [.05, .275, .50, .725, .95]


def rcs_basis(x, knots):
    k = len(knots); t = knots
    p3 = lambda v: np.maximum(v, 0) ** 3
    cols = [p3(x - t[j]) - p3(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
            + p3(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2]) for j in range(k - 2)]
    return np.column_stack(cols)


def wald(params, cov, idx):
    b = params[idx]; V = cov[np.ix_(idx, idx)]
    chi2 = float(b @ np.linalg.solve(V, b))
    return chi2, len(idx), float(stats.chi2.sf(chi2, len(idx)))


def fit(df, target):
    x = df[target].values.astype(float); y = df[OUTCOME].values
    knots = np.quantile(x, KNOT_PROBS)
    S = rcs_basis(x, knots)
    cov = [c for c in COVARIATES if c != target]
    X = np.column_stack([np.ones(len(x)), x, S, df[cov].values.astype(float)])
    m = sm.Logit(y, X).fit(disp=0, method="newton", maxiter=200)
    ns = S.shape[1]
    ov = wald(m.params, m.cov_params(), list(range(1, 2 + ns)))
    nl = wald(m.params, m.cov_params(), list(range(2, 2 + ns)))
    return dict(variable=target, knots=knots.round(3).tolist(),
                overall_chi2=ov[0], overall_df=ov[1], overall_p=ov[2],
                nonlin_chi2=nl[0], nonlin_df=nl[1], nonlin_p=nl[2], C=roc_auc_score(y, m.predict(X)))


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(DATA)
    tab = pd.DataFrame([fit(df, t) for t in TARGETS])
    tab.to_csv(f"{OUT}/spline_tests.csv", index=False)
    print(tab[["variable", "overall_chi2", "overall_p", "nonlin_chi2", "nonlin_p", "C"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
