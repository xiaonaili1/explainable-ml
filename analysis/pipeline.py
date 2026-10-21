"""Shared procedure: standardisation, LASSO selection and hyperparameter tuning fitted on the training data only."""
import warnings
import numpy as np
import pandas as pd
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

SEED = 42
N_TRIALS = 30
N_INNER_TUNE = 5
N_INNER_LASSO = 10
LASSO_CS = 20
ALGORITHMS = ["lr", "rf", "xgboost", "lightgbm", "svm", "catboost", "mlp"]
TREE_MODELS = ["xgboost", "rf", "lightgbm", "catboost"]
DATA = "data/analysis_dataset.csv"
OUTCOME = "outcome_si_binary"

CANDIDATES = [
    "gender_female", "gender_other", "education_years", "family_atmosphere", "sleep_hours", "screen_time",
    "screen_pct_gaming", "screen_pct_tv", "screen_pct_short_video", "screen_pct_chat", "screen_pct_reading",
    "screen_pct_social_media", "pa_level", "cpss_helplessness", "cpss_self_efficacy",
    "trauma_family_abuse", "trauma_gender_discrim", "trauma_school_bully", "trauma_cyberbully",
    "trauma_economic_discrim", "trauma_sexual_assault", "trauma_bereavement",
    "dx_adhd", "dx_depression", "dx_anxiety", "dx_ocd", "phq8_total", "gad10_total", "isi7_total", "age",
]
CONTINUOUS = [
    "screen_time", "sleep_hours", "education_years", "cpss_helplessness", "cpss_self_efficacy",
    "phq8_total", "gad10_total", "isi7_total", "screen_pct_gaming", "screen_pct_tv", "screen_pct_short_video",
    "screen_pct_chat", "screen_pct_reading", "screen_pct_social_media", "age",
]


def load_data():
    """Candidate predictors, binary outcome and the full dataset."""
    d = pd.read_csv(DATA)
    return d[CANDIDATES], d[OUTCOME].astype(int), d


def holdout_split():
    """Stratified 80/20 split with a fixed seed."""
    X, y, _ = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    return X, y, X_tr, X_te, y_tr, y_te


def scale_train_test(X_tr, X_te):
    cont = [c for c in CONTINUOUS if c in X_tr.columns]
    sc = StandardScaler().fit(X_tr[cont])
    X_tr_s, X_te_s = X_tr.copy(), X_te.copy()
    X_tr_s[cont] = sc.transform(X_tr[cont])
    X_te_s[cont] = sc.transform(X_te[cont])
    return X_tr_s, X_te_s, sc


def lasso_1se(X_tr_s, y_tr, seed=SEED):
    """L1 logistic regression with the penalty chosen by the one-standard-error rule on cross-validated AUC."""
    cv = StratifiedKFold(N_INNER_LASSO, shuffle=True, random_state=seed)
    m = LogisticRegressionCV(penalty="l1", solver="saga", class_weight="balanced",
                             cv=cv, Cs=LASSO_CS, scoring="roc_auc", refit=False,
                             max_iter=5000, random_state=seed).fit(X_tr_s, y_tr)
    scores = m.scores_[1]
    mean, se = scores.mean(0), scores.std(0, ddof=1) / np.sqrt(scores.shape[0])
    best = int(np.argmax(mean))
    c1se_idx = int(np.where(mean >= mean[best] - se[best])[0][0])
    C_1se = float(m.Cs_[c1se_idx])
    fit = LogisticRegression(penalty="l1", solver="saga", class_weight="balanced",
                             C=C_1se, max_iter=5000, random_state=seed).fit(X_tr_s, y_tr)
    coef = pd.Series(fit.coef_[0], index=X_tr_s.columns)
    selected = coef[coef != 0].abs().sort_values(ascending=False).index.tolist()
    return dict(selected=selected, coef=coef, C=C_1se)


def build(name, p, y_tr, seed=SEED):
    n_pos, n_neg = int((y_tr == 1).sum()), int((y_tr == 0).sum())
    spw = n_neg / n_pos if n_pos else 1.0
    if name == "lr":
        return LogisticRegression(C=p["C"], penalty="l2", solver="lbfgs", max_iter=1000,
                                  class_weight="balanced", random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                      min_samples_leaf=p["min_samples_leaf"],
                                      class_weight="balanced", random_state=seed, n_jobs=1)
    if name == "xgboost":
        return XGBClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                             learning_rate=p["learning_rate"], subsample=p["subsample"],
                             scale_pos_weight=spw, eval_metric="logloss", random_state=seed,
                             n_jobs=1, tree_method="exact", verbosity=0)
    if name == "lightgbm":
        return LGBMClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                              learning_rate=p["learning_rate"], num_leaves=p["num_leaves"],
                              class_weight="balanced", random_state=seed, n_jobs=1, verbose=-1)
    if name == "svm":
        return SVC(C=p["C"], kernel="rbf", gamma=p["gamma"],
                   class_weight="balanced", probability=True, random_state=seed)
    if name == "catboost":
        return CatBoostClassifier(iterations=p["iterations"], depth=p["depth"],
                                  learning_rate=p["learning_rate"], l2_leaf_reg=p["l2_leaf_reg"],
                                  random_seed=seed, logging_level="Silent", auto_class_weights="Balanced",
                                  allow_writing_files=False, thread_count=1)
    if name == "mlp":
        return Pipeline([("scaler", StandardScaler()),
                         ("mlp", MLPClassifier(hidden_layer_sizes=tuple(p["hidden_layer_sizes"]),
                                               activation=p["activation"], alpha=p["alpha"],
                                               learning_rate_init=p["lr"], max_iter=1000,
                                               early_stopping=True, n_iter_no_change=20,
                                               validation_fraction=0.15, random_state=seed))])
    raise KeyError(name)


def search_space(name, trial):
    if name == "lr":
        return {"C": trial.suggest_float("C", 0.01, 10, log=True)}
    if name == "rf":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20)}
    if name == "xgboost":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0)}
    if name == "lightgbm":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63)}
    if name == "svm":
        return {"C": trial.suggest_float("C", 0.1, 10, log=True),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"])}
    if name == "catboost":
        return {"iterations": trial.suggest_int("iterations", 100, 500),
                "depth": trial.suggest_int("depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10)}
    if name == "mlp":
        n_layers = trial.suggest_int("n_layers", 1, 3)
        return {"hidden_layer_sizes": tuple(trial.suggest_int(f"n_units_{i}", 32, 256) for i in range(n_layers)),
                "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True)}
    raise KeyError(name)


def tune(name, X_tr, y_tr, seed=SEED, n_trials=N_TRIALS):
    """Bayesian optimisation of the cross-validated AUC within the training data."""
    cv = StratifiedKFold(N_INNER_TUNE, shuffle=True, random_state=seed)
    folds = list(cv.split(X_tr, y_tr))

    def objective(trial):
        p = search_space(name, trial)
        aucs = []
        for tr, va in folds:
            m = build(name, p, y_tr.iloc[tr], seed).fit(X_tr.iloc[tr], y_tr.iloc[tr])
            aucs.append(roc_auc_score(y_tr.iloc[va], m.predict_proba(X_tr.iloc[va])[:, 1]))
        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = dict(study.best_params)
    if name == "mlp":
        bp["hidden_layer_sizes"] = tuple(bp[f"n_units_{i}"] for i in range(bp["n_layers"]))
    return bp, float(study.best_value)


def fit_fold(X_tr, y_tr, X_te, seed=SEED, algorithms=ALGORITHMS, n_trials=N_TRIALS):
    """Fit the whole procedure on the training data and score the test data once."""
    X_tr_s, X_te_s, sc = scale_train_test(X_tr, X_te)
    las = lasso_1se(X_tr_s, y_tr, seed)
    sel = las["selected"]
    print(f"    LASSO retained {len(sel)} of {X_tr.shape[1]} candidates")
    out = dict(selected=sel, lasso=las, scaler=sc, best_params={}, cv_auc={}, models={}, p_te={},
               X_tr_s=X_tr_s[sel], X_te_s=X_te_s[sel])
    for a in algorithms:
        bp, cv_auc = tune(a, X_tr_s[sel], y_tr, seed, n_trials)
        m = build(a, bp, y_tr, seed).fit(X_tr_s[sel], y_tr)
        out["best_params"][a], out["cv_auc"][a], out["models"][a] = bp, cv_auc, m
        out["p_te"][a] = m.predict_proba(X_te_s[sel])[:, 1]
        print(f"    {a:9s} cv_auc={cv_auc:.3f}")
    return out


def shap_values(model, X):
    """SHAP values of the positive class from a tree model."""
    import shap
    v = np.asarray(shap.TreeExplainer(model).shap_values(X))
    return v[:, :, 1] if v.ndim == 3 else v
