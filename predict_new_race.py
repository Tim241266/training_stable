import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from train_pipeline_v3_stable_patched import apply_preprocessor_fold, ensure_numpy, normalize_mongo_documents
from pymongo import MongoClient

# Load artifacts
preprocess_info = joblib.load("preprocess_info.joblib")
meta_clf = joblib.load("stacking_model.joblib")
final_bst = lgb.Booster(model_file="lgbm_booster_final.txt")

# MongoDB connection details
MONGO_URI = "mongodb://dbserver:27017"
MONGO_DB = "sportsfans"
MONGO_COLLECTION = "prospective_horses"  # Change to your new collection name

# Fetch new race data from MongoDB
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=50000, socketTimeoutMS=100000)
raw_docs = list(client[MONGO_DB][MONGO_COLLECTION].find({}))

# Normalize new race documents
X_new = normalize_mongo_documents(raw_docs)

# Add historical features
from train_pipeline_v3_stable_patched import add_historical_stats
X_new = add_historical_stats(X_new)

# Ensure all required columns are present and types are correct
for c in preprocess_info["high_card_cat"] + preprocess_info["low_card_cat"]:
    X_new[c] = X_new[c].fillna("UNKNOWN").astype(str)
for c in preprocess_info["num_features"]:
    X_new[c] = pd.to_numeric(X_new[c], errors="coerce").fillna(0)

# Preprocess new data
X_new_proc = apply_preprocessor_fold(
    X_new,
    preprocess_info,
    preprocess_info["num_features"],
    preprocess_info["high_card_cat"],
    preprocess_info["low_card_cat"]
)

# Get base model predictions
lgb_pred_new = np.asarray(final_bst.predict(X_new_proc.values, num_iteration=final_bst.best_iteration), dtype=float)
rf_pred_new = np.zeros_like(lgb_pred_new)  # If you want to use RandomForest, load and predict here
xgb_pred_new = np.zeros_like(lgb_pred_new) # If you want to use XGBoost, load and predict here

# Stack predictions as new features
X_stack_new = np.column_stack([lgb_pred_new, rf_pred_new, xgb_pred_new])

# Predict with stacking meta-model
stack_pred_proba = meta_clf.predict_proba(X_stack_new)[:, 1]
stack_pred_class = (stack_pred_proba > 0.5).astype(int)

# Output predictions
X_new["pred_proba"] = stack_pred_proba
X_new["pred_class"] = stack_pred_class
print(X_new[["pred_proba", "pred_class"]])

# Save predictions
X_new.to_csv("predictions_new_race.csv", index=False)
print("Predictions saved to predictions_new_race.csv")
