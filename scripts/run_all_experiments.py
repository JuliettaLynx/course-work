"""Все эксперименты ПР3 в одном воспроизводимом прогоне."""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, f1_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.features import (
    load_data, clean_data, make_pipeline,
    UserAvgRatingTransformer, add_features as add_features_fn,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)

# ----------------------------------------------------------------------
# Конфигурация
# ----------------------------------------------------------------------
RANDOM_STATE = 42
SAMPLE_SIZE  = 150_000
N_FOLDS      = 5
DIFF_THRESHOLD = 0.005

OUT_DIR    = Path("reports"); OUT_DIR.mkdir(exist_ok=True)
MODELS_DIR = Path("models");  MODELS_DIR.mkdir(exist_ok=True)

SCORING = {"roc_auc": "roc_auc",
           "pr_auc":  "average_precision",
           "f1":      "f1"}

# Признаки для абляции: "новые" (уходят в baseline=False) vs исходные
ORIGINAL_NUMERIC = ["average_rating", "num_pages",
                    "publication_year", "ratings_count"]
ORIGINAL_CAT     = ["is_ebook", "language_code"]

LOG = {}   # все результаты → all_experiments.json


# ----------------------------------------------------------------------
# Утилиты
# ----------------------------------------------------------------------
def to_python(obj):
    """Приводит numpy-типы к сериализуемым в JSON."""
    if isinstance(obj, dict):            return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):   return [to_python(v) for v in obj]
    if isinstance(obj, np.integer):      return int(obj)
    if isinstance(obj, np.floating):     return float(obj)
    if isinstance(obj, np.ndarray):      return obj.tolist()
    return obj


def cv_score(model_factory, X, y, groups, cv):
    """Возвращает (roc, pr, f1) как средние по фолдам."""
    roc, pr, f1 = [], [], []
    for tr, val in cv.split(X, y, groups):
        pipe = model_factory()
        pipe.fit(X.iloc[tr], y[tr])
        s = pipe.predict_proba(X.iloc[val])[:, 1]
        try:
            roc.append(roc_auc_score(y[val], s))
        except ValueError:
            roc.append(np.nan)
        pr.append(average_precision_score(y[val], s))
        f1.append(f1_score(y[val], (s >= 0.5).astype(int), zero_division=0))
    return (float(np.nanmean(roc)), float(np.nanstd(roc)),
            float(np.mean(pr)), float(np.mean(f1)))


def build_pipeline_no_new_features(model):
    """Pipeline без user-агрегатов и без feature engineering.

    Используется только для абляции признаков (раздел 7):
    baseline vs full. Для всех остальных экспериментов — make_pipeline
    из src.features.
    """
    num_t = Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler",  StandardScaler())])
    cat_t = Pipeline([("imputer", SimpleImputer(strategy="constant",
                                                fill_value="unknown")),
                      ("onehot",  OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", num_t, ORIGINAL_NUMERIC),
                              ("cat", cat_t, ORIGINAL_CAT)])
    return Pipeline([("prep", prep), ("model", model)])


# ----------------------------------------------------------------------
# 1. Данные (один раз)
# ----------------------------------------------------------------------
def load_all():
    print("[0/7] Загрузка данных...")
    t0 = time.time()
    df_full = clean_data(load_data())
    print(f"      full: {len(df_full):,} строк, "
          f"pos_rate={df_full['high_rating'].mean():.3f}")

    df_s = df_full.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
    X_s   = df_s.drop(columns=["high_rating"])
    y_s   = df_s["high_rating"].values
    grp_s = df_s["user_id"].values

    X_full = df_full.drop(columns=["high_rating"])
    y_full = df_full["high_rating"].values

    print(f"      sample: {len(df_s):,} строк, "
          f"users={len(np.unique(grp_s)):,}, "
          f"за {time.time()-t0:.1f}s")
    return X_s, y_s, grp_s, X_full, y_full


# ----------------------------------------------------------------------
# 2. fig1: L1 vs L2 по C
# ----------------------------------------------------------------------
def run_fig1(X, y, groups):
    print("\n[1/7] fig1: L1 vs L2 по C...")
    t0 = time.time()
    cv = GroupKFold(n_splits=N_FOLDS)
    C_values = [0.001, 0.01, 0.1, 1.0]
    roc_l1, roc_l2, nonzero = [], [], []

    for C in C_values:
        for l1_ratio, store in [(0.0, roc_l2), (1.0, roc_l1)]:
            factory = lambda r=l1_ratio, c=C: make_pipeline(
                LogisticRegression(C=c, l1_ratio=r, solver="liblinear",
                                max_iter=2000, random_state=RANDOM_STATE))
            roc, _, _, _ = cv_score(factory, X, y, groups, cv)
            store.append(roc)

            if l1_ratio == 1.0:
                pipe = factory(); pipe.fit(X, y)
                coef = pipe.named_steps["model"].coef_[0]
                nonzero.append(int((np.abs(coef) > 1e-8).sum()))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(C_values, roc_l1, "o-", label="L1")
    a1.plot(C_values, roc_l2, "s-", label="L2")
    a1.set_xscale("log"); a1.set_xlabel("C"); a1.set_ylabel("ROC-AUC")
    a1.set_title("Качество vs C"); a1.legend(); a1.grid(alpha=.3)
    a2.plot(C_values, nonzero, "o-", color="tab:orange")
    a2.set_xscale("log"); a2.set_xlabel("C")
    a2.set_ylabel("Ненулевых коэффициентов")
    a2.set_title("Разреживание L1"); a2.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_regularization.png", dpi=150); plt.close(fig)

    pd.DataFrame({"C": C_values, "roc_auc_l1": roc_l1,
                  "roc_auc_l2": roc_l2, "nonzero_l1": nonzero}
                 ).to_csv(OUT_DIR / "fig1_regularization.csv", index=False)
    LOG["fig1"] = {"C": C_values, "roc_l1": roc_l1,
                   "roc_l2": roc_l2, "nonzero_l1": nonzero}
    print(f"      L1: {[f'{v:.3f}' for v in roc_l1]}  ({time.time()-t0:.1f}s)")


# ----------------------------------------------------------------------
# 3. fig2: глубина дерева
# ----------------------------------------------------------------------
def run_fig2(X, y, groups):
    print("\n[2/7] fig2: глубина дерева...")
    t0 = time.time()
    cv = GroupKFold(n_splits=N_FOLDS)
    depths = list(range(1, 11))
    tr_s, val_s = [], []

    for d in depths:
        scores_tr, scores_val = [], []
        for tr, val in cv.split(X, y, groups):
            p = make_pipeline(DecisionTreeClassifier(
                max_depth=d, random_state=RANDOM_STATE))
            p.fit(X.iloc[tr], y[tr])
            scores_tr.append(roc_auc_score(
                y[tr], p.predict_proba(X.iloc[tr])[:, 1]))
            scores_val.append(roc_auc_score(
                y[val], p.predict_proba(X.iloc[val])[:, 1]))
        tr_s.append(float(np.mean(scores_tr)))
        val_s.append(float(np.mean(scores_val)))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(depths, tr_s, "o-", label="train")
    ax.plot(depths, val_s, "s-", label="val")
    ax.set_xlabel("max_depth"); ax.set_ylabel("ROC-AUC")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_tree_depth.png", dpi=150); plt.close(fig)

    LOG["fig2"] = {"depths": depths, "train": tr_s, "val": val_s}
    best_d = depths[int(np.argmax(val_s))]
    print(f"      best depth = {best_d}, val = {max(val_s):.4f}  "
          f"({time.time()-t0:.1f}s)")
    return best_d


# ----------------------------------------------------------------------
# 4. table2: абляция признаков
# ----------------------------------------------------------------------
def run_ablation(X, y, groups):
    print("\n[3/7] table2: абляция признаков...")
    t0 = time.time()
    cv = GroupKFold(n_splits=N_FOLDS)
    lr_l2 = lambda: LogisticRegression(C=0.001, l1_ratio=0.0, solver="lbfgs",
                                   max_iter=2000, random_state=RANDOM_STATE)
    lgbm_t = lambda: LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                    num_leaves=15, min_child_samples=20,
                                    reg_alpha=0.0, random_state=RANDOM_STATE,
                                    verbosity=-1)

    out = {}
    for name, model_f in [("LogReg-L2", lr_l2), ("LightGBM", lgbm_t)]:
        r_base, _, _, _ = cv_score(
            lambda mf=model_f: build_pipeline_no_new_features(mf()),
            X, y, groups, cv)
        r_full, _, _, _ = cv_score(
            lambda mf=model_f: make_pipeline(mf()), X, y, groups, cv)
        out[name] = {"baseline": r_base, "full": r_full,
                     "delta": r_full - r_base}
        print(f"      {name:<12} baseline={r_base:.4f}  "
              f"full={r_full:.4f}  Δ={r_full-r_base:+.4f}")

    LOG["ablation"] = out
    print(f"      ({time.time()-t0:.1f}s)")


# ----------------------------------------------------------------------
# 5. Подбор гиперпараметров
# ----------------------------------------------------------------------
def run_tuning(X, y, groups):
    print("\n[4/7] Подбор гиперпараметров...")
    cv = GroupKFold(n_splits=N_FOLDS)

    # LR (L1) — маленькая сетка, полный перебор
    t0 = time.time()
    lr_grid = {"model__C": [0.0001, 0.0005, 0.001, 0.005],
           "model__l1_ratio": [1.0],
           "model__solver": ["liblinear"],
           "model__max_iter": [2000]}
    s_lr = GridSearchCV(
        make_pipeline(LogisticRegression(random_state=RANDOM_STATE)),
        lr_grid, cv=cv, scoring=SCORING, refit="roc_auc", n_jobs=-1,
    ).fit(X, y, groups=groups)
    t_lr = time.time() - t0

    # LGBM — сокращённая сетка, полный перебор (24 комбинации)
    t0 = time.time()
    lgb_grid = {"model__n_estimators":      [200],
                "model__learning_rate":     [0.05, 0.1],
                "model__num_leaves":        [15, 31, 63],
                "model__min_child_samples": [20, 50],
                "model__reg_alpha":         [0.0, 0.1]}
    s_lgb = GridSearchCV(
        make_pipeline(LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1)),
        lgb_grid, cv=cv, scoring=SCORING, refit="roc_auc", n_jobs=-1,
    ).fit(X, y, groups=groups)
    t_lgb = time.time() - t0

    def summarize(s, name, elapsed):
        i = s.best_index_; r = s.cv_results_
        return {"model": name,
                "best_params": to_python(s.best_params_),
                "roc_auc": float(r["mean_test_roc_auc"][i]),
                "roc_auc_std": float(r["std_test_roc_auc"][i]),
                "pr_auc": float(r["mean_test_pr_auc"][i]),
                "f1": float(r["mean_test_f1"][i]),
                "time_sec": round(elapsed, 1),
                "n_combos": len(r["params"]),
                "n_folds": N_FOLDS}

    r_lr  = summarize(s_lr,  "LogReg-L1", t_lr)
    r_lgb = summarize(s_lgb, "LightGBM", t_lgb)

    diff = abs(r_lr["roc_auc"] - r_lgb["roc_auc"])
    if diff < DIFF_THRESHOLD:
        chosen = "LogReg-L1"
    else:
        chosen = max([r_lr, r_lgb], key=lambda r: r["roc_auc"])["model"]

    LOG["tuning"] = {"logreg": r_lr, "lgbm": r_lgb}
    LOG["choose_rule"] = {"threshold": DIFF_THRESHOLD,
                          "delta": round(diff, 4),
                          "chosen": chosen}

    print(f"      LR  C={s_lr.best_params_['model__C']}: "
          f"ROC-AUC={r_lr['roc_auc']:.4f} ({t_lr:.1f}s)")
    print(f"      LGB {s_lgb.best_params_}: "
          f"ROC-AUC={r_lgb['roc_auc']:.4f} ({t_lgb:.1f}s)")
    print(f"      Δ={diff:.4f} → выбрана {chosen}")
    return s_lr, s_lgb, chosen


# ----------------------------------------------------------------------
# 6. Сводная таблица (раздел 9)
# ----------------------------------------------------------------------
def run_comparison(X, y, groups, best_depth, lgbm_params):
    print("\n[5/7] table4: сводное сравнение...")
    t0 = time.time()
    cv = GroupKFold(n_splits=N_FOLDS)

    def evaluate(name, model_factory=None, baseline_fn=None):
        roc, pr, f1, rec = [], [], [], []
        t0 = time.time()
        for tr, val in cv.split(X, y, groups):
            if baseline_fn:
                s = baseline_fn(X.iloc[val])
            else:
                p = model_factory(); p.fit(X.iloc[tr], y[tr])
                s = p.predict_proba(X.iloc[val])[:, 1]
            preds = (s >= 0.5).astype(int)
            try:    roc.append(roc_auc_score(y[val], s))
            except: roc.append(np.nan)
            pr.append(average_precision_score(y[val], s))
            f1.append(f1_score(y[val], preds, zero_division=0))
            rec.append(recall_score(y[val], preds, zero_division=0))
        elapsed = round(time.time() - t0, 1)
        return {"model": name,
                "roc_auc": float(np.nanmean(roc)),
                "roc_auc_std": float(np.nanstd(roc)),
                "pr_auc": float(np.mean(pr)),
                "f1": float(np.mean(f1)),
                "recall": float(np.mean(rec)),
                "time_sec": elapsed}

    rows = [
        evaluate("Базовое: наиболее частый класс",
                 baseline_fn=lambda Xv: np.zeros(len(Xv))),
        evaluate("Базовое: правило rating≥4 & count≥100",
                 baseline_fn=lambda Xv: ((Xv["average_rating"] >= 4.0)
                                         & (Xv["ratings_count"] >= 100)
                                         ).astype(float).values),
        evaluate("LogReg-L1 (C=0.0005) — итоговая", lambda: make_pipeline(
            LogisticRegression(C=0.0005, l1_ratio=1.0, solver="liblinear",
                            max_iter=2000, random_state=RANDOM_STATE))),
        evaluate("LogReg-L2 (C=0.001)", lambda: make_pipeline(
            LogisticRegression(C=0.001, l1_ratio=0.0, solver="lbfgs",
                            max_iter=2000, random_state=RANDOM_STATE))),
        evaluate(f"Decision Tree (depth={best_depth})", lambda: make_pipeline(
            DecisionTreeClassifier(max_depth=best_depth,
                                   random_state=RANDOM_STATE))),
        evaluate("Random Forest (100)", lambda: make_pipeline(
            RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                   random_state=RANDOM_STATE))),
        evaluate("LightGBM (tuned)", lambda: make_pipeline(LGBMClassifier(
            **lgbm_params, random_state=RANDOM_STATE, verbosity=-1))),
    ]

    df_out = pd.DataFrame(rows)
    df_out_display = df_out.copy()
    df_out_display["roc_auc"] = df_out.apply(
        lambda r: f"{r['roc_auc']:.4f} ± {r['roc_auc_std']:.4f}", axis=1)
    df_out_display = df_out_display[
        ["model", "roc_auc", "pr_auc", "f1", "recall", "time_sec"]
    ].rename(columns={"model": "Модель", "roc_auc": "ROC-AUC",
                    "pr_auc": "PR-AUC", "f1": "F1@0.5",
                    "recall": "Полнота@0.5", "time_sec": "Время CV, с"})
    df_out_display.to_csv(OUT_DIR / "table_comparison.csv", index=False)
    print(df_out_display.to_string(index=False))
    print(f"      ({time.time()-t0:.1f}s)")
    LOG["comparison"] = rows


# ----------------------------------------------------------------------
# 7. Финальная модель
# ----------------------------------------------------------------------
def fit_final(X_full, y_full, s_lr, s_lgb, chosen):
    print(f"\n[6/7] Финальная модель ({chosen}) на полном train...")
    t0 = time.time()
    best = s_lr if chosen == "LogReg-L1" else s_lgb
    final = best.best_estimator_
    final.fit(X_full, y_full)
    joblib.dump(final, MODELS_DIR / "model_v1.joblib")
    print(f"      X_full={X_full.shape}  ({time.time()-t0:.1f}s)")
    return final, best


def save_meta(X_full, s_lr, s_lgb, chosen):
    print("\n[7/7] Сохранение метаданных...")
    best = s_lr if chosen == "LogReg-L1" else s_lgb
    meta = {
        "date": pd.Timestamp.now().isoformat(timespec="seconds"),
        "algorithm": chosen,
        "params": to_python(best.best_params_),
        "roc_auc_cv_sample": float(best.best_score_),
        "roc_auc_cv_std": float(
            best.cv_results_["std_test_roc_auc"][best.best_index_]),
        "pr_auc_cv": float(
            best.cv_results_["mean_test_pr_auc"][best.best_index_]),
        "f1_cv": float(
            best.cv_results_["mean_test_f1"][best.best_index_]),
        "cv_scheme": f"GroupKFold(n_splits={N_FOLDS}) by user_id",
        "sample_size_search": SAMPLE_SIZE,
        "sample_size_final_fit": int(X_full.shape[0]),
        "features_numeric": NUMERIC_FEATURES,
        "features_categorical": CATEGORICAL_FEATURES,
        "random_state": RANDOM_STATE,
        "search_results": LOG["tuning"],
        "choose_rule": LOG["choose_rule"],
        "all_experiments": to_python(LOG),
        "notes": ("Все эксперименты выполнены на единой подвыборке "
                  f"{SAMPLE_SIZE} строк (GroupKFold({N_FOLDS}) по user_id). "
                  "Финальная модель переобучена на полном train (2001–2013). "
                  "Тестовая выборка (2014) не использовалась."),
    }
    (MODELS_DIR / "model_v1.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (OUT_DIR / "all_experiments.json").write_text(
        json.dumps(to_python(LOG), ensure_ascii=False, indent=2),
        encoding="utf-8")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    t_start = time.time()
    print("=" * 60)
    print("ПР3: все эксперименты, единая схема")
    print("=" * 60)

    X, y, groups, X_full, y_full = load_all()

    run_fig1(X, y, groups)
    best_depth = run_fig2(X, y, groups)
    run_ablation(X, y, groups)
    s_lr, s_lgb, chosen = run_tuning(X, y, groups)

    # Параметры LGBM для сравнения — берём лучшее из поиска
    lgbm_best = s_lgb.best_params_
    lgbm_params = {k.replace("model__", ""): v
                   for k, v in lgbm_best.items()}
    run_comparison(X, y, groups, best_depth, lgbm_params)

    fit_final(X_full, y_full, s_lr, s_lgb, chosen)
    save_meta(X_full, s_lr, s_lgb, chosen)

    print("\n" + "=" * 60)
    print(f"Готово за {time.time() - t_start:.1f}s")
    print("=" * 60)
    print("Артефакты:")
    print("  models/model_v1.joblib")
    print("  models/model_v1.json")
    print("  reports/fig1_regularization.{png,csv}")
    print("  reports/fig2_tree_depth.png")
    print("  reports/table_comparison.csv")
    print("  reports/all_experiments.json")


if __name__ == "__main__":
    main()