# ================================================================================
# utils.py — ICU Sepsis Mortality Prediction
# ================================================================================

import os
import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics         import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.exception import CustomException


# ================================================================================
# HELPER : F2 Score
# ================================================================================

def f2_score(precision, recall):
    """F2 weights recall 2x — critical for ICU mortality prediction."""
    if (4 * precision + recall) == 0:
        return 0.0
    return 5 * (precision * recall) / ((4 * precision) + recall)


# ================================================================================
# SAVE OBJECT
# ================================================================================

def save_object(file_path: str, obj):
    """Serialize and save any Python object to disk using pickle."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


# ================================================================================
# LOAD OBJECT
# ================================================================================

def load_object(file_path: str):
    """Load a serialized Python object from disk."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


# ================================================================================
# EVALUATE MODELS — Classification (replaces regression r2_score version)
# ================================================================================

def evaluate_models(X_train, y_train, X_test, y_test, models: dict, param: dict):
    """
    Train and evaluate multiple classifiers using RandomizedSearchCV.
    Returns a report dict with AUC score per model.

    Changes from original utils.py:
      - GridSearchCV      → RandomizedSearchCV  (faster)
      - r2_score          → roc_auc_score       (classification metric)
      - KFold             → StratifiedKFold     (preserves class ratio)
    """
    try:
        report = {}
        cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in models.items():
            para = param.get(name, {})

            # --- Tune if param grid provided ---
            if para:
                search = RandomizedSearchCV(
                    estimator           = model,
                    param_distributions = para,
                    n_iter              = 15,
                    scoring             = 'roc_auc',
                    cv                  = cv,
                    n_jobs              = -1,
                    random_state        = 42,
                    verbose             = 0
                )
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
            else:
                # No param grid — train directly
                model.fit(X_train, y_train)
                best_model = model

            # --- Predict probabilities ---
            y_prob_train = best_model.predict_proba(X_train)[:, 1]
            y_prob_test  = best_model.predict_proba(X_test)[:,  1]
            y_pred_test  = (y_prob_test >= 0.5).astype(int)

            # --- Metrics ---
            auc_train  = roc_auc_score(y_train, y_prob_train)
            auc_test   = roc_auc_score(y_test,  y_prob_test)
            prec       = precision_score(y_test, y_pred_test, zero_division=0)
            rec        = recall_score(y_test,    y_pred_test, zero_division=0)
            f1         = f1_score(y_test,        y_pred_test, zero_division=0)
            f2         = f2_score(prec, rec)

            report[name] = {
                'model'      : best_model,
                'Train AUC'  : auc_train,
                'Test AUC'   : auc_test,
                'F1'         : f1,
                'Precision'  : prec,
                'Recall'     : rec,
                'F2'         : f2,
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)


# ================================================================================
# FIND BEST THRESHOLD — Maximize F2 (Recall-weighted)
# ================================================================================

def find_best_threshold(y_true, y_prob):
    """
    Scan thresholds 0.01 → 0.99 and return the one
    that maximizes F2 score (recall-weighted for medical use).
    """
    try:
        thresholds  = np.arange(0.01, 1.00, 0.01)
        best_f2     = -1
        best_thresh = 0.5

        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            prec   = precision_score(y_true, y_pred, zero_division=0)
            rec    = recall_score(y_true,    y_pred, zero_division=0)
            f2     = f2_score(prec, rec)
            if f2 > best_f2:
                best_f2     = f2
                best_thresh = t

        return float(best_thresh), float(best_f2)

    except Exception as e:
        raise CustomException(e, sys)