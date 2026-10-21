"""Figures 1, 2, S1 and S2 from the held-out evaluation (results/figures/)."""
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, shap
from sklearn.metrics import roc_curve, brier_score_loss
from analysis.pipeline import load_data, TREE_MODELS

HOLDOUT = "results/holdout"
OUT = "results/figures"
LABELS = {"phq8_total": "PHQ-8 total", "cpss_helplessness": "Helplessness (CPSS)", "dx_depression": "Depression diagnosis",
          "gad10_total": "GAD-10 total", "dx_adhd": "ADHD diagnosis", "cpss_self_efficacy": "Self-efficacy (CPSS, reverse-scored)",
          "screen_pct_tv": "Screen time: TV share", "education_years": "Education (years)",
          "screen_pct_gaming": "Screen time: gaming share", "screen_pct_chat": "Screen time: chat share",
          "isi7_total": "ISI-7 total", "family_atmosphere": "Family atmosphere", "age": "Age (years)"}
NAMES = {"xgboost": "XGBoost", "rf": "Random Forest", "mlp": "MLP", "lr": "Logistic Regression",
         "catboost": "CatBoost", "svm": "SVM", "lightgbm": "LightGBM"}


def main():
    os.makedirs(OUT, exist_ok=True)
    P = pd.read_parquet(f"{HOLDOUT}/test_predictions.parquet"); y = P.y_true.values
    met = pd.read_csv(f"{HOLDOUT}/metrics.csv").sort_values("auc", ascending=False)
    z = np.load(f"{HOLDOUT}/shap_full_refit.npz", allow_pickle=True); sv = z["shap_values"]; feats = list(z["features"])
    top_tree = next(a for a in met.algorithm if a in TREE_MODELS)

    # Figure 1: ROC curves
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=1, label="Chance")
    for a, auc in zip(met.algorithm, met.auc):
        fpr, tpr, _ = roc_curve(y, P[f"prob_{a}"]); ax.plot(fpr, tpr, lw=1.2, label=f"{NAMES[a]} (AUC = {auc:.3f})")
    ax.set_xlabel("1 − Specificity"); ax.set_ylabel("Sensitivity"); ax.legend(fontsize=8, loc="lower right", frameon=False)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/Figure_1.pdf"); fig.savefig(f"{OUT}/Figure_1.png", dpi=300); plt.close()

    # Figure 2: SHAP beeswarm and mean absolute SHAP values
    X, _, _ = load_data()
    d = X[feats].copy(); d.columns = [LABELS.get(f, f) for f in feats]
    imp = pd.Series(np.abs(sv).mean(0), index=d.columns).sort_values()
    fig = plt.figure(figsize=(13, 5.4)); fig.add_subplot(1, 2, 1)
    shap.summary_plot(sv, d, show=False, plot_size=None, max_display=len(feats)); ax1 = plt.gca()
    ax1.text(-0.35, 1.02, "A", transform=ax1.transAxes, fontweight="bold", fontsize=13)
    ax2 = fig.add_subplot(1, 2, 2); ax2.barh(imp.index, imp.values, color="#1f77b4"); ax2.set_xlabel("Mean |SHAP value|")
    ax2.text(-0.4, 1.02, "B", transform=ax2.transAxes, fontweight="bold", fontsize=13)
    for s in ["top", "right"]: ax2.spines[s].set_visible(False)
    fig.tight_layout(w_pad=6); fig.savefig(f"{OUT}/Figure_2.pdf"); fig.savefig(f"{OUT}/Figure_2.png", dpi=300); plt.close()

    # Figure S1: calibration curves on the test set
    fig, ax = plt.subplots(figsize=(5, 5)); ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    for a in [top_tree, "rf", "lr"]:
        p = P[f"prob_{a}"].values; g = pd.DataFrame({"p": p, "y": y}).groupby(pd.qcut(p, 10), observed=True).mean()
        ax.plot(g.p, g.y, "o-", lw=1, ms=4, label=f"{NAMES[a]} (Brier = {brier_score_loss(y, p):.3f})")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency"); ax.legend(fontsize=8, frameon=False)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/Figure_S1.pdf"); fig.savefig(f"{OUT}/Figure_S1.png", dpi=300); plt.close()

    # Figure S2: decision curves
    th = np.arange(0.01, 0.61, 0.01); prev = y.mean(); n = len(y)
    def net_benefit(p):
        return np.array([((p >= t) & (y == 1)).sum() / n - ((p >= t) & (y == 0)).sum() / n * t / (1 - t) for t in th])
    treat_all = prev - (1 - prev) * th / (1 - th)
    fig, ax = plt.subplots(figsize=(5.5, 4.5)); ax.plot(th, treat_all, "k--", lw=1, label="Treat all"); ax.axhline(0, c="grey", lw=1, label="Treat none")
    for a in [top_tree, "lr"]: ax.plot(th, net_benefit(P[f"prob_{a}"].values), lw=1.2, label=NAMES[a])
    ax.set_ylim(-0.05, 0.65); ax.set_xlim(0, 0.6); ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit"); ax.legend(fontsize=8, frameon=False)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/Figure_S2.pdf"); fig.savefig(f"{OUT}/Figure_S2.png", dpi=300); plt.close()
    for a in [top_tree, "lr"]:
        v = net_benefit(P[f"prob_{a}"].values); ok = th[(v > treat_all + 0.005) & (v > 0)]
        print(f"{NAMES[a]}: net benefit above treat-all from {ok.min():.2f} to {ok.max():.2f}")


if __name__ == "__main__":
    main()
