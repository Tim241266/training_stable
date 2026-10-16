# predict_new_race_blend.py
"""
Prediction script for blending ensemble (LightGBM, RandomForest, XGBoost + LogisticRegression meta-model).
Fetches new race data from MongoDB, preprocesses, adds historical stats, and outputs predictions.
"""
import os
import joblib
import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, PyMongoError
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# --- Config ---
MONGO_URI = "mongodb://dbserver:27017"
MONGO_DB = "sportsfans"
MONGO_COLLECTION = "prospective_horses"

PREPROCESS_PATH = "preprocess_info.joblib"
LGBM_MODEL_PATH = "lgbm_booster_final.txt"
BLEND_MODEL_PATH = "blending_model.joblib"

# --- Helper: ensure numpy ---
def ensure_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    elif hasattr(x, "toarray"):
        return np.asarray(x.toarray())
    else:
        return np.asarray(x)

# --- Load preprocess info ---
preprocess_info = joblib.load(PREPROCESS_PATH)
num_features = preprocess_info["num_features"]
high_card_cat = preprocess_info["high_card_cat"]
low_card_cat = preprocess_info["low_card_cat"]

# --- Mongo fetch ---
def fetch_new_race_docs():
    try:
        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=50000,
            socketTimeoutMS=100000
        )
        client.server_info()
        docs = list(client[MONGO_DB][MONGO_COLLECTION].find({}))
        return docs
    except ServerSelectionTimeoutError as e:
        print(f"MongoDB connection error: {e}")
        return []
    except PyMongoError as e:
        print(f"MongoDB error: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

# --- Normalization and historical stats ---
from train_pipeline_v3_stable_patched import normalize_mongo_documents, add_historical_stats

def preprocess_new_race(docs):
    df = normalize_mongo_documents(docs)
    df = add_historical_stats(df)
    for c in high_card_cat + low_card_cat:
        df[c] = df[c].fillna("UNKNOWN").astype(str)
    df[num_features] = df[num_features].apply(pd.to_numeric, errors="coerce")
    return df

def preprocess_new_race_with_history(docs):
    # Load history table from training
    history_df = preprocess_info["history_table"]
    # Normalize new race docs
    df_new = normalize_mongo_documents(docs)
    # Concatenate history and new races
    df_all = pd.concat([history_df, df_new], ignore_index=True)
    # Add historical stats to the combined DataFrame
    df_all = add_historical_stats(df_all)
    # Extract only the new races (the last N rows)
    df_new_hist = df_all.iloc[-len(df_new):].reset_index(drop=True)
    # Ensure correct types
    for c in high_card_cat + low_card_cat:
        df_new_hist[c] = df_new_hist[c].fillna("UNKNOWN").astype(str)
    df_new_hist[num_features] = df_new_hist[num_features].apply(pd.to_numeric, errors="coerce")
    return df_new_hist

# --- Apply preprocess ---
def apply_preprocessor(X, info):
    Xp = X.copy()
    for c in num_features:
        Xp[c] = pd.to_numeric(Xp[c], errors="coerce").fillna(info["medians"].get(c, 0.0))
    X_num = pd.DataFrame(info["scaler"].transform(Xp[num_features]), columns=num_features, index=Xp.index)
    if info["target_encoder"] and high_card_cat:
        X_high = info["target_encoder"].transform(Xp[high_card_cat].astype(str))
        X_high = pd.DataFrame(X_high, columns=[f"{c}_te" for c in high_card_cat], index=Xp.index)
    else:
        X_high = pd.DataFrame(index=Xp.index)
    if info["onehot"] and low_card_cat:
        ohe_arr = info["onehot"].transform(Xp[low_card_cat].astype(str))
        ohe_df = pd.DataFrame(ensure_numpy(ohe_arr), columns=info["onehot_cols"], index=Xp.index)
    else:
        ohe_df = pd.DataFrame(index=Xp.index)
    final = pd.concat([X_num, X_high, ohe_df], axis=1)
    final = final.loc[:, sorted(final.columns)]
    return final

# --- Load models ---
print("Loading LightGBM...")
lgbm_bst = lgb.Booster(model_file=LGBM_MODEL_PATH)
print("Loading blending meta-model...")
blend_meta_clf = joblib.load(BLEND_MODEL_PATH)
print("Loading base RandomForest and XGBoost...")
rf_blend = RandomForestClassifier(n_estimators=200, random_state=42)
xgb_blend = xgb.XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='logloss')
# NOTE: These base models must be re-trained or loaded from disk if saved. Here, we fit on history table.
# For prediction, fit on preprocess_info["history_table"]
history_df = preprocess_info["history_table"]
X_hist = apply_preprocessor(history_df[num_features + high_card_cat + low_card_cat], preprocess_info)
y_hist = history_df["is_win"]
rf_blend.fit(X_hist, y_hist)
xgb_blend.fit(X_hist, y_hist)

# --- Prediction ---
def predict_blend():
    docs = fetch_new_race_docs()
    if not docs:
        print("No new race data found.")
        return
    df_new = preprocess_new_race_with_history(docs)
    X_new = df_new[num_features + high_card_cat + low_card_cat]
    X_new_proc = apply_preprocessor(X_new, preprocess_info)
    # Base model predictions
    lgb_pred = np.asarray(lgbm_bst.predict(X_new_proc.values), dtype=float)
    rf_pred = np.asarray(rf_blend.predict_proba(X_new_proc)[:, 1], dtype=float)
    xgb_pred = np.asarray(xgb_blend.predict_proba(X_new_proc)[:, 1], dtype=float)
    X_meta = np.column_stack([lgb_pred, rf_pred, xgb_pred])
    # Blending meta-model prediction
    blend_pred_proba = blend_meta_clf.predict_proba(X_meta)[:, 1]
    blend_pred_class = (blend_pred_proba > 0.7).astype(int)
    df_new["blend_pred_proba"] = blend_pred_proba
    df_new["blend_pred_class"] = blend_pred_class
    # Output
    df_new.to_csv("predictions_new_race_blend.csv", index=False)
    print("Predictions saved to predictions_new_race_blend.csv")
    print(df_new[["horse", "blend_pred_proba", "blend_pred_class"]].head())

if __name__ == "__main__":
    predict_blend()
