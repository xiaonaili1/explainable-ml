"""Figure 3: restricted cubic spline dose-response curves for depression severity, perceived helplessness and anxiety
severity, each adjusted for the other covariates, with bootstrap 95% confidence bands."""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from analysis.pipeline import DATA, OUTCOME, SEED
from analysis.splines import COVARIATES, rcs_basis

OUT = "results/figures"
N_BOOT = 1000
KNOT_PERCENTILES = [5, 27.5, 50, 72.5, 95]
PANELS = [("phq8_total", "PHQ-8 Depression Severity"), ("cpss_helplessness", "CPSS Perceived Helplessness"),
          ("gad10_total", "GAD-10 Anxiety Severity")]
COLOR = "#2E75B6"
plt.rcParams.update({"font.family": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight"})


def estimator():
    # a very light ridge keeps the bootstrap fits well conditioned without changing the estimates materially
    return LogisticRegression(C=1e4, max_iter=1000, solver="lbfgs")


def spline_curve(x, y, covariates):
    knots = np.percentile(x, KNOT_PERCENTILES)
    X_full = np.column_stack([x.reshape(-1, 1), rcs_basis(x, knots), covariates])
    model = estimator().fit(X_full, y)
    x_grid = np.linspace(np.percentile(x, 2), np.percentile(x, 98), 200)
    X_grid = np.column_stack([x_grid.reshape(-1, 1), rcs_basis(x_grid, knots),
                              np.tile(covariates.mean(axis=0), (len(x_grid), 1))])
    pred = model.predict_proba(X_grid)[:, 1]
    boot = np.zeros((N_BOOT, len(x_grid)))
    for b in range(N_BOOT):
        idx = resample(np.arange(len(y)), random_state=SEED + b)
        boot[b] = estimator().fit(X_full[idx], y[idx]).predict_proba(X_grid)[:, 1]
    return x_grid, pred, np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(DATA)
    y = df[OUTCOME].values
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, letter, (name, label) in zip(axes, "ABC", PANELS):
        covars = [c for c in COVARIATES if c != name]
        x = df[name].values
        x_grid, pred, lo, hi = spline_curve(x, y, df[covars].values)
        ax.plot(x_grid, pred, color=COLOR, linewidth=2.5)
        ax.fill_between(x_grid, lo, hi, alpha=0.2, color=COLOR)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)
        ax.set_ylim(0.0, 1.0)
        ax.plot(x, np.full_like(x, 0.02), "|", color=COLOR, alpha=0.08, markersize=6)
        ax.set_xlabel(label, fontsize=12)
        ax.set_title(letter, fontsize=14, fontweight="bold", loc="left")
    axes[0].set_ylabel("Predicted probability of PHQ-9 item 9 endorsement", fontsize=12)
    plt.tight_layout()
    fig.savefig(f"{OUT}/Figure_3.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT}/Figure_3.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
