# train_pipeline_v3_stable.py
"""
Stable training pipeline (Option A).

Features:
 - Robust race header parsing
 - normalize Mongo documents -> DataFrame
 - add chronological historical features (horse/jockey/trainer wins/runs/winrate)
 - fold-aware preprocessing:
     * numeric -> StandardScaler
     * high-cardinality cats -> category_encoders.TargetEncoder (fit on train fold)
     * low-cardinality cats -> OneHotEncoder (fit on train fold)
 - Optuna tuning: per-fold preprocessing + LightGBM training with early stopping
 - Final fit on full train using best params
 - Saves: lgbm_booster_final.txt, preprocess_info.joblib
"""

from datetime import datetime
import re
import os
import joblib
import numpy as np
import pandas as pd
from pymongo import MongoClient
from typing import Dict, Any
import optuna
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import category_encoders as ce
import warnings

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# -----------------------
# Parsing utilities
# -----------------------
WEEKDAYS = r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"

def clean_race_string(s: str) -> str:
    if not s:
        return ""
    s = str(s)
    s = s.replace("\u00a0", " ").replace("\u200b", "").replace("\u202f", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_race_header(s: str):
    """
    Return tuple (race_time_str, track, race_date or None)
    Handles many variants like:
   - "Race Result 17:55 BATH Friday 01 August"
   - "Race Result 23:00 TURF PARADISE (USA) 28 January 2025"
    """
    if not s:
        return None, None, None
    s = clean_race_string(s)
    s = s.replace("Race Result", "").strip()

    # time
    tm = re.search(r"\b(\d{1,2}:\d{2})\b", s)
    race_time = tm.group(1) if tm else None
    s_wo_time = re.sub(r"\b\d{1,2}:\d{2}\b", "", s).strip()

    race_date = None
    date_match = None

    # Pattern A: Weekday DD Month YYYY
    pat_a = re.search(rf"\b{WEEKDAYS}\b\s+(\d{{1,2}})\s+([A-Za-z]+)\s+(\d{{4}})", s_wo_time, flags=re.IGNORECASE)
    if pat_a:
        day, mon, year = pat_a.group(2), pat_a.group(3), pat_a.group(4)
        for fmt in ("%d %B %Y", "%d %b %Y"):
            try:
                race_date = datetime.strptime(f"{day} {mon} {year}", fmt)
                break
            except:
                race_date = None
        date_match = pat_a

    # Pattern B: Weekday DD Month (no year)
    if race_date is None:
        pat_b = re.search(rf"\b{WEEKDAYS}\b\s+(\d{{1,2}})\s+([A-Za-z]+)", s_wo_time, flags=re.IGNORECASE)
        if pat_b:
            day, mon = pat_b.group(1), pat_b.group(2)
            for yr in (datetime.now().year, datetime.now().year - 1):
                for fmt in ("%d %B %Y", "%d %b %Y"):
                    try:
                        race_date = datetime.strptime(f"{day} {mon} {yr}", fmt)
                        break
                    except:
                        race_date = None
                if race_date:
                    break
            date_match = pat_b if race_date else date_match

    # Pattern C: DD Month YYYY
    if race_date is None:
        pat_c = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", s_wo_time, flags=re.IGNORECASE)
        if pat_c:
            day, mon, year = pat_c.group(1), pat_c.group(2), pat_c.group(3)
            for fmt in ("%d %B %Y", "%d %b %Y"):
                try:
                    race_date = datetime.strptime(f"{day} {mon} {year}", fmt)
                    break
                except:
                    race_date = None
            date_match = pat_c if race_date else date_match

    # Pattern D: DD Month (no year)
    if race_date is None:
        pat_d = re.search(r"(\d{1,2})\s+([A-Za-z]+)", s_wo_time, flags=re.IGNORECASE)
        if pat_d:
            day, mon = pat_d.group(1), pat_d.group(2)
            for yr in (datetime.now().year, datetime.now().year - 1):
                for fmt in ("%d %B %Y", "%d %b %Y"):
                    try:
                        race_date = datetime.strptime(f"{day} {mon} {yr}", fmt)
                        break
                    except:
                        race_date = None
                if race_date:
                    break
            date_match = pat_d if race_date else date_match

    track = s_wo_time
    if date_match:
        track = track.replace(date_match.group(0), "").strip()
    track = re.sub(r"\s+", " ", track).strip()
    track = track.upper() if track else None

    return race_time, track, race_date

# -----------------------
# Normalize Mongo documents
# -----------------------
def normalize_mongo_documents(raw_docs):
    rows = []
    for doc in raw_docs:
        rows.append({
            "race": doc.get("race"),
            "distance": doc.get("distance"),
            "prize": doc.get("prize"),
            "ageRules": doc.get("ageRules"),
            "raceType": doc.get("raceType"),
            "surface": doc.get("surface"),
            "going": doc.get("going"),
            "horse": doc.get("horse"),
            "jockey": doc.get("jockey"),
            "trainer": doc.get("trainer"),
            "age": doc.get("age"),
            "weight": doc.get("weight"),
            "isp": doc.get("isp"),
            "position": doc.get("position"),
            "_raw": doc
        })
    df = pd.DataFrame(rows)

    cleaned = df["race"].fillna("").astype(str).apply(clean_race_string)
    df["race_clean"] = cleaned
    df["race_time_str"], df["track"], df["race_date"] = zip(*cleaned.map(parse_race_header))

    # simple parsers
    def parse_distance(d):
        if not d: return np.nan
        s = str(d).lower()
        m = re.search(r"(\d+)\s*m\b", s)
        if m and "f" not in s and "y" not in s:
            return float(m.group(1))
        miles = furl = yards = 0
        mm = re.search(r"(\d+)\s*m(?:iles?)?", s)
        ff = re.search(r"(\d+)\s*f(?:urlong)?s?", s)
        yy = re.search(r"(\d+)\s*y", s)
        if mm: miles = int(mm.group(1))
        if ff: furl = int(ff.group(1))
        if yy: yards = int(yy.group(1))
        meters = miles*1609.344 + furl*201.168 + yards*0.9144
        return meters if meters>0 else np.nan

    def parse_weight(w):
        if not w: return np.nan
        s = str(w).lower()
        m = re.search(r"(\d+)-(\d+)", s)
        if m:
            stones, lbs = int(m.group(1)), int(m.group(2))
            return (stones*14 + lbs) * 0.453592
        m2 = re.search(r"(\d+)\s*(lb|lbs)?", s)
        if m2:
            return int(m2.group(1)) * 0.453592
        return np.nan

    def parse_odds(o):
        if not o: return np.nan
        s = str(o).strip().upper()
        try: return float(s)
        except: pass
        m = re.match(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", s)
        if m: return float(m.group(1))/float(m.group(2)) + 1.0
        m2 = re.match(r"(\d+)-(\d+)", s)
        if m2: return float(m2.group(1))/float(m2.group(2)) + 1.0
        if s in ("EVS","EVENS"): return 2.0
        return np.nan

    df["distance_m"] = df["distance"].apply(parse_distance)
    df["weight_kg"] = df["weight"].apply(parse_weight)
    df["odds_decimal"] = df["isp"].apply(parse_odds)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    df["is_win"] = (df["position"] == 1).astype(int)

    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce").dt.normalize()
    df["race_id_key"] = (df["race_date"].astype(str).fillna("") + "_" +
                         df["track"].fillna("").astype(str) + "_" +
                         df["race_time_str"].fillna("").astype(str)).str.strip("_")
    df["race_id"] = pd.factorize(df["race_id_key"])[0]

    return df

# -----------------------
# Historical features (chronological)
# -----------------------
def add_historical_stats(df: pd.DataFrame) -> pd.DataFrame:
    # assume df has race_date and race_id (normalized)
    df = df.sort_values(["race_date", "race_id"]).reset_index(drop=False)
    for key in ["horse", "jockey", "trainer"]:
        df[f"{key}_runs_before"] = df.groupby(key).cumcount()
        df[f"{key}_wins_before"] = df.groupby(key)["is_win"].transform(lambda s: s.shift().cumsum().fillna(0).astype(int))
        df[f"{key}_winrate"] = df[f"{key}_wins_before"] / df[f"{key}_runs_before"].replace(0, np.nan)
        df[f"{key}_winrate"] = df[f"{key}_winrate"].fillna(0.0)
    df = df.set_index("index").sort_index().reset_index(drop=True)
    for c in [f"{k}_runs_before" for k in ["horse","jockey","trainer"]] + [f"{k}_wins_before" for k in ["horse","jockey","trainer"]]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df

# -----------------------
# Fold-aware preprocess (fit on train fold only)
# -----------------------
def fit_preprocessor_fold(X_tr: pd.DataFrame, y_tr: pd.Series,
                          num_features, high_card_cat, low_card_cat) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    # numeric scaler
    scaler = StandardScaler()
    # fill numeric NaNs with median per column (train)
    Xnum = X_tr[num_features].copy()
    medians = {}
    for c in num_features:
        med = float(Xnum[c].median(skipna=True)) if Xnum[c].notna().any() else 0.0
        medians[c] = med
        Xnum[c] = pd.to_numeric(Xnum[c], errors="coerce").fillna(med)
    scaler.fit(Xnum)
    info["scaler"] = scaler
    info["medians"] = medians

    # target encoder for high-cardinality categories
    if len(high_card_cat) > 0:
        te = ce.TargetEncoder(cols=high_card_cat, smoothing=0.3)
        te.fit(X_tr[high_card_cat].astype(str), y_tr)
    else:
        te = None
    info["target_encoder"] = te

    # one-hot encoder for low-cardinality categories
    if len(low_card_cat) > 0:
        # sklearn >=1.4 uses sparse_output, older versions use sparse
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        ohe.fit(X_tr[low_card_cat].astype(str))
        ohe_cols = list(ohe.get_feature_names_out(low_card_cat))
    else:
        ohe = None
        ohe_cols = []
    info["onehot"] = ohe
    info["onehot_cols"] = ohe_cols

    return info

def apply_preprocessor_fold(X: pd.DataFrame, info: Dict[str, Any],
                            num_features, high_card_cat, low_card_cat) -> pd.DataFrame:
    # numeric
    Xp = X.copy()
    for c in num_features:
        Xp[c] = pd.to_numeric(Xp[c], errors="coerce").fillna(info["medians"].get(c, 0.0))
    X_num = pd.DataFrame(info["scaler"].transform(Xp[num_features]), columns=num_features, index=Xp.index)

    # target encode high-card
    if info["target_encoder"] is not None and len(high_card_cat) > 0:
        X_high = info["target_encoder"].transform(Xp[high_card_cat].astype(str))
        # TargetEncoder returns same column names
        X_high = pd.DataFrame(X_high, columns=high_card_cat, index=Xp.index)
        # rename to indicate encoded
        X_high = X_high.rename(columns={c: f"{c}_te" for c in high_card_cat})
    else:
        X_high = pd.DataFrame(index=Xp.index)

    # one-hot low-card
    if info["onehot"] is not None and len(low_card_cat) > 0:
        ohe_arr = info["onehot"].transform(Xp[low_card_cat].astype(str))
        ohe_df = pd.DataFrame(ohe_arr, columns=info["onehot_cols"], index=Xp.index)
    else:
        ohe_df = pd.DataFrame(index=Xp.index)

    final = pd.concat([X_num, X_high, ohe_df], axis=1)
    # deterministic ordering
    final = final.loc[:, sorted(final.columns)]
    return final

# -----------------------
# Optuna tuning (per-fold preprocess)
# -----------------------
def tune_lgbm(preprocess_cfg, X: pd.DataFrame, y: pd.Series,
              num_features, high_card_cat, low_card_cat,
              n_trials=7, cv_folds=5, n_jobs=1):
    """
    preprocess_cfg unused here but kept for API symmetry.
    X must be DataFrame, y Series.
    """
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        # sample params
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.3),
            "n_jobs": n_jobs,
            "seed": RANDOM_STATE
        }

        fold_aucs = []
        for train_idx, valid_idx in cv.split(X, y):
            X_tr_raw, X_val_raw = X.iloc[train_idx], X.iloc[valid_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

            # fit preprocess on train fold
            info = fit_preprocessor_fold(X_tr_raw, y_tr, num_features, high_card_cat, low_card_cat)
            X_tr = apply_preprocessor_fold(X_tr_raw, info, num_features, high_card_cat, low_card_cat)
            X_val = apply_preprocessor_fold(X_val_raw, info, num_features, high_card_cat, low_card_cat)

            # LightGBM dataset
            dtrain = lgb.Dataset(X_tr.values, label=y_tr.values)
            dval = lgb.Dataset(X_val.values, label=y_val.values, reference=dtrain)

            # small n_estimators for tuning for speed
            num_round = 800
            bst = lgb.train(
                params,
                dtrain,
                num_boost_round=num_round,
                valid_sets=[dtrain, dval],
                valid_names=["train","valid"],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                # early_stopping_rounds=50,
                # verbose_eval=False
            )

            # predict on validation using best iteration
            iters = bst.best_iteration or num_round
            preds = bst.predict(X_val.values, num_iteration=iters)
            # fold_auc = roc_auc_score(y_val.values, preds)

            # force dense numpy array — covers any edge case
            if not isinstance(preds, np.ndarray):
                preds = np.asarray(preds)

            fold_auc = roc_auc_score(y_val.values, preds.astype(float))
            fold_aucs.append(fold_auc)

        # return mean AUC across folds
        return float(np.mean(fold_aucs))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("Optuna best params:", study.best_params)
    print("Best CV AUC:", study.best_value)
    return study.best_params

def ensure_numpy(x):
    return x if isinstance(x, np.ndarray) else np.asarray(x)

# -----------------------
# Main training flow
# -----------------------
def main(
    mongo_uri="mongodb://localhost:27017",
    mongo_db="fastapi_results",
    mongo_col="tresults",
    n_trials=7,
    cv_folds=5,
    n_jobs=4
):
    # 1) load raw docs
    client = MongoClient(mongo_uri)
    raw_docs = list(client[mongo_db][mongo_col].find({}))
    df = normalize_mongo_documents(raw_docs)
    print("1) After Mongo load:", len(df))

    # 2) drop rows missing race_date (can't order)
    print("Null counts (head):")
    print(df.isna().sum())
    df = df.dropna(subset=["race_date"])
    df.loc[:, "track"] = df["track"].fillna("UNKNOWN")
    df.loc[:, "distance_m"] = df["distance_m"].fillna(df["distance_m"].median())
    df.loc[:, "weight_kg"] = df["weight_kg"].fillna(df["weight_kg"].median())
    df.loc[:, "odds_decimal"] = df["odds_decimal"].fillna(df["odds_decimal"].median())
    df.loc[:, "age"] = df["age"].fillna(df["age"].median())
    print("2) After dropna:", len(df))

    # 3) historical stats
    df = add_historical_stats(df)
    print("3) After add_historical_stats:", len(df))

    # 4) define features
    num_features = ["distance_m", "weight_kg", "age", "odds_decimal", "horse_winrate", "jockey_winrate", "trainer_winrate"]
    high_card_cat = ["horse", "jockey", "trainer"]
    low_card_cat = ["surface", "going", "raceType", "track"]

    # ensure categorical strings
    for c in high_card_cat + low_card_cat:
        df.loc[:, c] = df[c].fillna("UNKNOWN").astype(str)

    # 5) chronological split (80/20)
    df = df.sort_values("race_date").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    print("Train rows:", len(df_train), "Test rows:", len(df_test))

    X_train_all = df_train[num_features + high_card_cat + low_card_cat].reset_index(drop=True)
    y_train_all = df_train["is_win"].reset_index(drop=True)

    X_test_raw = df_test[num_features + high_card_cat + low_card_cat].reset_index(drop=True)
    y_test = df_test["is_win"].reset_index(drop=True)

    # 6) compute scale_pos_weight
    pos = int(y_train_all.sum())
    neg = len(y_train_all) - pos
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0
    print(f"pos={pos}, neg={neg}, scale_pos_weight={scale_pos_weight:.3f}")

    # 7) run Optuna tuning
    best_params = tune_lgbm(None, X_train_all, y_train_all, num_features, high_card_cat, low_card_cat,
                            n_trials=n_trials, cv_folds=cv_folds, n_jobs=n_jobs)

    # attach fixed params and scale_pos_weight
    best_params.update({
        "scale_pos_weight": scale_pos_weight,
        "n_jobs": n_jobs,
        "seed": RANDOM_STATE,
        "verbosity": -1
    })
    best_params.setdefault("objective", "binary")
    best_params.setdefault("metric", "auc")
    # choose final n_estimators (higher)
    final_num_round = 2000

    # 8) Fit preprocess on full training set (fitted objects saved to preprocess_info)
    preprocess_info = {}
    preprocess_info["num_features"] = num_features
    preprocess_info["high_card_cat"] = high_card_cat
    preprocess_info["low_card_cat"] = low_card_cat
    preprocess_info["history_table"] = df_train.copy()  # store history for downstream additions

    # Fit final preprocessors on full training data
    final_info = fit_preprocessor_fold(X_train_all, y_train_all, num_features, high_card_cat, low_card_cat)
    preprocess_info["scaler"] = final_info["scaler"]
    preprocess_info["medians"] = final_info["medians"]
    preprocess_info["target_encoder"] = final_info["target_encoder"]
    preprocess_info["onehot"] = final_info["onehot"]
    preprocess_info["onehot_cols"] = final_info["onehot_cols"]

    # 9) Transform entire train and test using fitted final_info
    X_train_proc = apply_preprocessor_fold(X_train_all, final_info, num_features, high_card_cat, low_card_cat)
    X_test_proc = apply_preprocessor_fold(X_test_raw, final_info, num_features, high_card_cat, low_card_cat)

    # 10) final LightGBM train (with early stopping on test)
    dtrain = lgb.Dataset(X_train_proc.values, label=y_train_all.values)
    dvalid = lgb.Dataset(X_test_proc.values, label=y_test.values, reference=dtrain)

    print("Training final LightGBM with best params...")
    final_bst = lgb.train(
        best_params,
        dtrain,
        num_boost_round=final_num_round,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True)]
    )

    # 11) evaluation on test
    pred_proba = final_bst.predict(X_test_proc.values, num_iteration=final_bst.best_iteration or final_num_round)
    # pred_class = (pred_proba > 0.5).astype(int)

    # force into a dense numpy array
    if not isinstance(pred_proba, np.ndarray):
        pred_proba = np.asarray(pred_proba)

    pred_proba = pred_proba.astype(float)

    pred_class = (pred_proba > 0.5).astype(int)

    # print("AUC:", roc_auc_score(y_test.values, pred_proba))
    y_true = y_test.to_numpy(dtype=float)

    if not isinstance(pred_proba, np.ndarray):
        pred_proba = np.asarray(pred_proba, dtype=float)

    auc = roc_auc_score(y_true, pred_proba)
    print("AUC:", auc)

    y_true = ensure_numpy(y_test).astype(float)
    pred_proba = ensure_numpy(pred_proba).astype(float)

    print("Log Loss:", log_loss(y_true, pred_proba))
    # print("Log Loss:", log_loss(y_test.values, pred_proba))
    # print("Accuracy:", accuracy_score(y_test.values, pred_class))

    y_true = y_test.to_numpy(dtype=int)
    pred_class = np.asarray(pred_class, dtype=int)

    print("Accuracy:", accuracy_score(y_true, pred_class))

    # 12) Save artifacts
    model_path = "lgbm_booster_final.txt"
    preprocess_path = "preprocess_info.joblib"
    final_bst.save_model(model_path)
    joblib.dump(preprocess_info, preprocess_path)
    print(f"Saved model -> {model_path}")
    print(f"Saved preprocess info -> {preprocess_path}")

    # try feature importance
    try:
        imp = final_bst.feature_importance(importance_type="gain")
        feat_names = list(X_train_proc.columns)
        imp_df = pd.DataFrame({"feature": feat_names, "importance": imp}).sort_values("importance", ascending=False)
        print("Top features:\n", imp_df.head(20))
    except Exception as e:
        print("Feature importance error:", e)

    return final_bst, preprocess_info, df

if __name__ == "__main__":
    bst, prep_info, df_all = main(
        mongo_uri="mongodb://localhost:27017",
        mongo_db="fastapi_results",
        mongo_col="tresults",
        n_trials=7,      # tune as needed
        cv_folds=5,
        n_jobs=4
    )
