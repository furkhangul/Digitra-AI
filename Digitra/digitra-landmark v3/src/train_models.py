"""Model comparison + Optuna hyperparameter optimization.

Strict protocol:
  * StandardScaler (where needed) is fit on TRAIN only.
  * Hyperparameter tuning uses the VALIDATION person (P8) exclusively.
  * The TEST persons (P9, P10) are NEVER seen during training or tuning.
  * The test set is evaluated exactly ONCE, after the best model is chosen.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import joblib
import warnings
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from utils import ARTIFACTS_DIR, MODELS_DIR, REPORTS_DIR, setup_logging
from feature_engineering import build_feature_matrix, feature_names

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_AUG = 1
SEED = 42
SVM_CAP = 5000  # SVC is O(n^2); cap training samples for SVM only
# Per-model Optuna trial budgets (kept modest for a reasonable runtime).
TRIALS = {"svm": 6, "catboost": 6, "xgboost": 6,
          "lightgbm": 6, "mlp": 6, "extratrees": 6}
# Global wall-clock budget for the whole tuning stage (seconds). If exceeded,
# we stop starting new models and proceed with whichever finished (resumable).
TUNING_BUDGET_S = 3000


# ----------------------------- data ----------------------------------------
def get_data(force: bool = False):
    logger = setup_logging()
    cache = ARTIFACTS_DIR / f"features_cache_naug{N_AUG}.npz"
    if cache.exists() and not force:
        logger.info("Loaded cached features: %s", cache)
        d = np.load(cache, allow_pickle=True)
        return (d["X_train"], d["y_train"], d["X_val"], d["y_val"],
                d["X_test"], d["y_test"], list(d["classes"]))

    df = pd.read_parquet(ARTIFACTS_DIR / "splits.parquet")
    rng = np.random.default_rng(SEED)

    train_df = df[df.split == "train"]
    val_df = df[df.split == "val"]
    test_df = df[df.split == "test"]

    logger.info("Building TRAIN features (aug=%d) ...", N_AUG)
    X_train, y_train = build_feature_matrix(train_df, augment=True,
                                           n_aug=N_AUG, rng=rng)
    logger.info("Building VAL features ...")
    X_val, y_val = build_feature_matrix(val_df)
    logger.info("Building TEST features ...")
    X_test, y_test = build_feature_matrix(test_df)

    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_val = le.transform(y_val)
    y_test = le.transform(y_test)

    np.savez(cache, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test, classes=le.classes_)
    logger.info("Feature cache saved. shapes: train=%s val=%s test=%s",
                X_train.shape, X_val.shape, X_test.shape)
    return X_train, y_train, X_val, y_val, X_test, y_test, list(le.classes_)


# --------------------------- model factory ---------------------------------
def make_model(name, p, n_classes):
    if name == "svm":
        # probability=False: avoids the internal 5-fold CV, dramatically
        # faster. SVM is excluded from the soft-vote ensemble automatically
        # (no predict_proba) but still a valid single-model candidate.
        return SVC(C=p["C"], gamma=p["gamma"], kernel="rbf",
                   probability=False, random_state=SEED, cache_size=2000)
    if name == "catboost":
        return cb.CatBoostClassifier(iterations=p["iterations"],
                                     learning_rate=p["lr"], depth=p["depth"],
                                     l2_leaf_reg=p["l2"],
                                     loss_function="MultiClass",
                                     verbose=False, random_seed=SEED)
    if name == "xgboost":
        return xgb.XGBClassifier(n_estimators=p["n_estimators"],
                                 max_depth=p["max_depth"],
                                 learning_rate=p["lr"],
                                 subsample=p["subsample"],
                                 colsample_bytree=p["colsample"],
                                 reg_lambda=p["reg_lambda"],
                                 n_jobs=-1, random_state=SEED,
                                 eval_metric="mlogloss")
    if name == "lightgbm":
        return lgb.LGBMClassifier(n_estimators=p["n_estimators"],
                                  num_leaves=p["num_leaves"],
                                  learning_rate=p["lr"],
                                  subsample=p["subsample"],
                                  colsample_bytree=p["colsample"],
                                  reg_lambda=p["reg_lambda"],
                                  min_child_samples=p["min_child"],
                                  n_jobs=-1, random_state=SEED,
                                  verbose=-1)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=p["hidden"],
                             alpha=p["alpha"],
                             learning_rate_init=p["lr_init"],
                             max_iter=400, early_stopping=True,
                             validation_fraction=0.1, n_iter_no_change=15,
                             random_state=SEED)
    if name == "extratrees":
        return ExtraTreesClassifier(n_estimators=p["n_estimators"],
                                    max_depth=p["max_depth"],
                                    min_samples_leaf=p["min_samples_leaf"],
                                    max_features=p["max_features"],
                                    n_jobs=-1, random_state=SEED)
    raise ValueError(name)


def sample_params(name, trial: optuna.trial.Trial):
    if name == "svm":
        return {"C": trial.suggest_float("C", 0.5, 100.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-4, 1e-1, log=True)}
    if name == "catboost":
        return {"iterations": trial.suggest_int("iterations", 200, 500, step=100),
                "lr": trial.suggest_float("lr", 0.01, 0.2, log=True),
                "depth": trial.suggest_int("depth", 4, 8),
                "l2": trial.suggest_float("l2", 1.0, 10.0, log=True)}
    if name == "xgboost":
        return {"n_estimators": trial.suggest_int("n_estimators", 200, 500, step=100),
                "max_depth": trial.suggest_int("max_depth", 4, 9),
                "lr": trial.suggest_float("lr", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample": trial.suggest_float("colsample", 0.6, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0, log=True)}
    if name == "lightgbm":
        return {"n_estimators": trial.suggest_int("n_estimators", 200, 500, step=100),
                "num_leaves": trial.suggest_int("num_leaves", 20, 96),
                "lr": trial.suggest_float("lr", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample": trial.suggest_float("colsample", 0.6, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0, log=True),
                "min_child": trial.suggest_int("min_child", 5, 40)}
    if name == "mlp":
        return {"hidden": trial.suggest_categorical(
                    "hidden", [(256, 128), (256, 128, 64), (512, 256)]),
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
                "lr_init": trial.suggest_float("lr_init", 1e-4, 1e-2, log=True)}
    if name == "extratrees":
        return {"n_estimators": trial.suggest_int("n_estimators", 200, 500, step=100),
                "max_depth": trial.suggest_int("max_depth", 10, 40),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_float("max_features", 0.3, 1.0)}
    raise ValueError(name)


SCALED = {"svm", "mlp"}


def objective(name, Xtr, ytr, Xval, yval, n_classes):
    def _obj(trial):
        p = sample_params(name, trial)
        Xt, yt = Xtr, ytr
        if name == "svm" and Xt.shape[0] > SVM_CAP:
            idx = np.random.RandomState(SEED).choice(Xt.shape[0], SVM_CAP, replace=False)
            Xt, yt = Xt[idx], yt[idx]
        if name in SCALED:
            sc = StandardScaler().fit(Xt)
            Xt2, Xv2 = sc.transform(Xt), sc.transform(Xval)
        else:
            Xt2, Xv2 = Xt, Xval
        m = make_model(name, p, n_classes)
        m.fit(Xt2, yt)
        pred = m.predict(Xv2)
        return -f1_score(yval, pred, average="macro")
    return _obj


def tune(name, Xtr, ytr, Xval, yval, n_classes, n_trials):
    logger = setup_logging()
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective(name, Xtr, ytr, Xval, yval, n_classes),
                   n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    logger.info("[%s] best val macroF1=%.4f params=%s",
                name, -study.best_value, best)
    return best, -study.best_value


def train_final(name, params, Xtr, ytr, Xval, yval, n_classes):
    """Train final model on full train; return (model, scaler_or_None, val_pred)."""
    sc = None
    if name in SCALED:
        sc = StandardScaler().fit(Xtr)
        Xt2, Xv2 = sc.transform(Xtr), sc.transform(Xval)
    else:
        Xt2, Xv2 = Xtr, Xval
    m = make_model(name, params, n_classes)
    m.fit(Xt2, ytr)
    return m, sc, m.predict(Xv2), (m.predict_proba(Xv2) if hasattr(m, "predict_proba") else None)


def run(force_features: bool = False, n_trials: int = 20,
        ensemble: bool = True, force: bool = False):
    logger = setup_logging()
    X_train, y_train, X_val, y_val, X_test, y_test, classes = get_data(force_features)
    n_classes = len(classes)

    names = ["svm", "catboost", "xgboost", "lightgbm", "mlp", "extratrees"]
    trials = {n: TRIALS.get(n, max(8, n_trials)) for n in names}

    progress_path = ARTIFACTS_DIR / "pipeline_progress.json"
    progress = {}
    if progress_path.exists() and not force:
        try:
            progress = json.load(open(progress_path))
        except Exception:
            progress = {}

    trained = {}
    t_start = time.time()

    def _reload(name):
        m = joblib.load(MODELS_DIR / f"{name}_cand.pkl")
        scp = MODELS_DIR / f"{name}_cand_scaler.pkl"
        sc = joblib.load(scp) if scp.exists() else None
        Xv = sc.transform(X_val) if sc is not None else X_val
        vp = m.predict_proba(Xv) if hasattr(m, "predict_proba") else None
        return m, sc, progress[name]["params"], vp

    for name in names:
        if name in progress and not force:
            try:
                trained[name] = _reload(name)
                logger.info("[%s] resumed from checkpoint val_macroF1=%.4f",
                            name, progress[name]["val_macro_f1"])
                continue
            except Exception:
                pass
        if (time.time() - t_start) > TUNING_BUDGET_S and len(trained) >= 1:
            logger.info("Tuning budget reached; proceeding with %d model(s): %s",
                        len(trained), list(trained.keys()))
            break
        t0 = time.time()
        best_p, val_f1 = tune(name, X_train, y_train, X_val, y_val,
                              n_classes, trials[name])
        m, sc, val_pred, val_proba = train_final(name, best_p, X_train, y_train,
                                                 X_val, y_val, n_classes)
        val_acc = accuracy_score(y_val, val_pred)
        rep = classification_report(y_val, val_pred, output_dict=True, zero_division=0)
        val_prec = rep["macro avg"]["precision"]
        val_rec = rep["macro avg"]["recall"]
        trained[name] = (m, sc, best_p, val_proba)
        joblib.dump(m, MODELS_DIR / f"{name}_cand.pkl")
        if sc is not None:
            joblib.dump(sc, MODELS_DIR / f"{name}_cand_scaler.pkl")
        progress[name] = {"params": best_p, "val_accuracy": val_acc,
                         "val_macro_f1": val_f1, "val_macro_precision": val_prec,
                         "val_macro_recall": val_rec, "time_s": time.time() - t0}
        json.dump(progress, open(progress_path, "w"), indent=2)
        logger.info("[%s] val_acc=%.4f val_macroF1=%.4f (%.1fs)",
                    name, val_acc, val_f1, time.time() - t0)

    if not trained:
        raise RuntimeError("No model completed tuning (check budget / data).")

    comparison = [{
        "model": n, "val_accuracy": progress[n]["val_accuracy"],
        "val_macro_precision": progress[n]["val_macro_precision"],
        "val_macro_recall": progress[n]["val_macro_recall"],
        "val_macro_f1": progress[n]["val_macro_f1"],
        "params": progress[n]["params"], "time_s": progress[n]["time_s"],
    } for n in trained]
    comp_df = pd.DataFrame(comparison).sort_values(
        "val_macro_f1", ascending=False).reset_index(drop=True)
    comp_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    logger.info("Model comparison:\n%s",
                comp_df[["model", "val_accuracy", "val_macro_f1"]].to_string(index=False))

    best_single = comp_df.iloc[0]
    use_ensemble = False
    proba_models = [(n, trained[n][3]) for n in trained if trained[n][3] is not None]
    if ensemble and len(proba_models) >= 2:
        avg = np.mean([p for _, p in proba_models], axis=0)
        ens_pred = avg.argmax(axis=1)
        ens_f1 = f1_score(y_val, ens_pred, average="macro")
        ens_acc = accuracy_score(y_val, ens_pred)
        logger.info("Ensemble (soft-vote) val_acc=%.4f val_macroF1=%.4f", ens_acc, ens_f1)
        if ens_f1 > best_single["val_macro_f1"] + 1e-4:
            use_ensemble = True
            logger.info("Ensemble selected (improves validation).")

    if use_ensemble:
        test_probas = []
        for n, (m, sc, _, _) in trained.items():
            if hasattr(m, "predict_proba"):
                Xte = sc.transform(X_test) if sc is not None else X_test
                test_probas.append(m.predict_proba(Xte))
        test_pred = np.mean(test_probas, axis=0).argmax(axis=1)
        chosen = "ensemble"
    else:
        name = best_single["model"]
        m, sc, _, _ = trained[name]
        Xte = sc.transform(X_test) if sc is not None else X_test
        test_pred = m.predict(Xte)
        chosen = name

    if use_ensemble:
        tr_probas = []
        for n, (m, sc, _, _) in trained.items():
            if hasattr(m, "predict_proba"):
                Xtr_t = sc.transform(X_train) if sc is not None else X_train
                tr_probas.append(m.predict_proba(Xtr_t))
        train_pred = np.mean(tr_probas, axis=0).argmax(axis=1)
    else:
        name = best_single["model"]
        m, sc, _, _ = trained[name]
        Xtr_t = sc.transform(X_train) if sc is not None else X_train
        train_pred = m.predict(Xtr_t)
    train_acc = accuracy_score(y_train, train_pred)

    test_acc = accuracy_score(y_test, test_pred)
    test_prec = f1_score(y_test, test_pred, average="macro", zero_division=0)
    rep_test = classification_report(y_test, test_pred, output_dict=True, zero_division=0)
    test_macro_p = rep_test["macro avg"]["precision"]
    test_macro_r = rep_test["macro avg"]["recall"]
    test_macro_f1 = rep_test["macro avg"]["f1-score"]
    logger.info("FINAL TEST (model=%s) acc=%.4f macroP=%.4f macroR=%.4f macroF1=%.4f",
                chosen, test_acc, test_macro_p, test_macro_r, test_macro_f1)

    if use_ensemble:
        for n, (m, sc, p, _) in trained.items():
            if hasattr(m, "predict_proba"):
                joblib.dump(m, MODELS_DIR / f"ensemble_{n}.pkl")
                if sc is not None:
                    joblib.dump(sc, MODELS_DIR / f"ensemble_{n}_scaler.pkl")
        meta = {"model": "ensemble",
                "members": [n for n, (m, sc, p, pr) in trained.items() if pr is not None],
                "label_classes": classes}
    else:
        name = best_single["model"]
        m, sc, p, _ = trained[name]
        joblib.dump(m, MODELS_DIR / "best_model.pkl")
        if sc is not None:
            joblib.dump(sc, MODELS_DIR / "scaler.pkl")
        le = LabelEncoder(); le.classes_ = np.array(classes)
        joblib.dump(le, MODELS_DIR / "label_encoder.pkl")
        feat_cfg = {"feature_names": feature_names(),
                    "n_features": len(feature_names()),
                    "model": name, "params": p, "scaled": name in SCALED}
        with open(MODELS_DIR / "feature_config.json", "w") as f:
            json.dump(feat_cfg, f, indent=2)
        meta = {"model": name, "params": p, "scaled": name in SCALED,
                "label_classes": classes}

    with open(MODELS_DIR / "pipeline_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "comparison": comp_df,
        "chosen": chosen,
        "best_single": best_single.to_dict(),
        "test_acc": test_acc, "test_macro_p": test_macro_p,
        "test_macro_r": test_macro_r, "test_macro_f1": test_macro_f1,
        "val_acc": best_single["val_accuracy"], "val_macro_f1": best_single["val_macro_f1"],
        "train_acc": train_acc, "train_val_gap": train_acc - best_single["val_accuracy"],
        "y_test": y_test, "test_pred": test_pred, "classes": classes,
        "rep_test": rep_test,
    }


if __name__ == "__main__":
    run()
