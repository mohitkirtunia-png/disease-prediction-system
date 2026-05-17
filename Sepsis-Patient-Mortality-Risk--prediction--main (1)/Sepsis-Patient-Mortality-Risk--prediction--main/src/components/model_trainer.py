import os
import sys

# --- Fix Python path ---
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import numpy as np
from dataclasses import dataclass
from xgboost     import XGBClassifier

from sklearn.metrics         import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, classification_report
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.exception import CustomException
from src.logger    import logging
from src.utils     import save_object, f2_score


@dataclass
class ModelTrainerConfig:
    trained_model_file_path : str = os.path.join('artifacts', 'model.pkl')
    threshold_file_path     : str = os.path.join('artifacts', 'threshold.txt')
    config_file_path        : str = os.path.join('artifacts', 'model_config.json')


def evaluate_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    prec   = precision_score(y_true, y_pred, zero_division=0)
    rec    = recall_score(y_true,    y_pred, zero_division=0)
    return {
        'threshold' : threshold,
        'auc'       : roc_auc_score(y_true, y_prob),
        'accuracy'  : accuracy_score(y_true, y_pred),
        'f1'        : f1_score(y_true, y_pred, zero_division=0),
        'precision' : prec,
        'recall'    : rec,
        'f2'        : f2_score(prec, rec),
    }


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def find_best_threshold(self, y_true, y_prob,
                             min_precision: float = 0.35):
        thresholds   = np.arange(0.05, 0.95, 0.01)
        best_f2      = -1
        best_thresh  = 0.5
        best_metrics = None

        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            prec   = precision_score(y_true, y_pred, zero_division=0)
            rec    = recall_score(y_true,    y_pred, zero_division=0)
            f2     = f2_score(prec, rec)

            if prec >= min_precision and f2 > best_f2:
                best_f2      = f2
                best_thresh  = t
                best_metrics = {
                    'threshold': t,
                    'precision': prec,
                    'recall'   : rec,
                    'f2'       : f2
                }

        if best_metrics:
            logging.info(
                f"Best threshold : {best_thresh:.2f} | "
                f"F2: {best_metrics['f2']:.4f} | "
                f"Recall: {best_metrics['recall']:.4f} | "
                f"Precision: {best_metrics['precision']:.4f}"
            )
        else:
            logging.warning(
                f"No threshold met min_precision={min_precision}. "
                f"Falling back to 0.5"
            )
            best_thresh = 0.5

        return float(best_thresh), float(best_f2)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Model Training started")

            X_train = train_array[:, :-1]
            y_train = train_array[:,  -1]
            X_test  = test_array[:,  :-1]
            y_test  = test_array[:,   -1]

            logging.info(
                f"X_train: {X_train.shape} | X_test: {X_test.shape}")
            logging.info(
                f"Train mortality: {y_train.mean()*100:.2f}% | "
                f"Test  mortality: {y_test.mean()*100:.2f}%"
            )

            param_grid = {
                'n_estimators'     : [50, 100, 200],
                'learning_rate'    : [0.01, 0.1, 0.2],
                'max_depth'        : [3, 5, 7],
                'subsample'        : [0.7, 0.8, 1.0],
                'colsample_bytree' : [0.7, 0.8, 1.0],
                'min_child_weight' : [1, 3, 5],
                'reg_alpha'        : [0, 0.1, 0.5],
                'reg_lambda'       : [1, 1.5, 2.0],
            }

            xgb_base = XGBClassifier(
                random_state      = 42,
                use_label_encoder = False,
                eval_metric       = 'logloss',
                n_jobs            = -1
            )

            cv = StratifiedKFold(
                n_splits=5, shuffle=True, random_state=42)

            logging.info("RandomizedSearchCV started")
            search = RandomizedSearchCV(
                estimator           = xgb_base,
                param_distributions = param_grid,
                n_iter              = 20,
                scoring             = 'roc_auc',
                cv                  = cv,
                n_jobs              = -1,
                random_state        = 42,
                verbose             = 1
            )
            search.fit(X_train, y_train)

            best_model  = search.best_estimator_
            best_params = search.best_params_
            cv_auc      = search.best_score_

            logging.info(f"Best params : {best_params}")
            logging.info(f"CV AUC      : {cv_auc:.4f}")

            y_prob_train  = best_model.predict_proba(X_train)[:, 1]
            y_prob_test   = best_model.predict_proba(X_test)[:,  1]

            train_metrics = evaluate_at_threshold(
                y_train, y_prob_train, 0.5)
            test_metrics  = evaluate_at_threshold(
                y_test,  y_prob_test,  0.5)

            logging.info(
                f"Train (0.50) | AUC:{train_metrics['auc']:.4f} | "
                f"Recall:{train_metrics['recall']:.4f} | "
                f"F2:{train_metrics['f2']:.4f}"
            )
            logging.info(
                f"Test  (0.50) | AUC:{test_metrics['auc']:.4f} | "
                f"Recall:{test_metrics['recall']:.4f} | "
                f"F2:{test_metrics['f2']:.4f}"
            )

            logging.info("Threshold tuning (min_precision=0.35)")
            best_threshold, best_f2 = self.find_best_threshold(
                y_test, y_prob_test, min_precision=0.35)

            tuned = evaluate_at_threshold(
                y_test, y_prob_test, best_threshold)

            logging.info(
                f"Tuned ({best_threshold:.2f}) | "
                f"AUC:{tuned['auc']:.4f} | "
                f"Recall:{tuned['recall']:.4f} | "
                f"Precision:{tuned['precision']:.4f} | "
                f"F2:{tuned['f2']:.4f}"
            )

            if tuned['auc'] < 0.75:
                raise CustomException(
                    f"AUC {tuned['auc']:.4f} below minimum 0.75", sys)

            print("\n" + "="*65)
            print("MODEL TRAINING COMPLETE — SUMMARY")
            print("="*65)
            print(f"  Model          : XGBClassifier")
            print(f"  CV AUC         : {cv_auc:.4f}")
            print(f"  Best Params    : {best_params}")
            print(f"\n  Default Threshold (0.50):")
            print(f"    AUC          : {test_metrics['auc']:.4f}")
            print(f"    Accuracy     : {test_metrics['accuracy']:.4f}")
            print(f"    Recall       : {test_metrics['recall']:.4f}")
            print(f"    Precision    : {test_metrics['precision']:.4f}")
            print(f"    F1           : {test_metrics['f1']:.4f}")
            print(f"    F2           : {test_metrics['f2']:.4f}")
            print(f"\n  Tuned Threshold ({best_threshold:.2f}) "
                  f"[min_precision=0.35]:")
            print(f"    AUC          : {tuned['auc']:.4f}")
            print(f"    Accuracy     : {tuned['accuracy']:.4f}")
            print(f"    Recall       : {tuned['recall']:.4f}")
            print(f"    Precision    : {tuned['precision']:.4f}")
            print(f"    F1           : {tuned['f1']:.4f}")
            print(f"    F2           : {tuned['f2']:.4f}")
            print("="*65)

            y_pred_final = (y_prob_test >= best_threshold).astype(int)
            print(classification_report(
                y_test, y_pred_final,
                target_names=['Survived (0)', 'Died (1)']
            ))

            # --- Save model ---
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj       = best_model
            )

            # --- Save threshold ---
            os.makedirs(
                os.path.dirname(
                    self.model_trainer_config.threshold_file_path),
                exist_ok=True
            )
            with open(self.model_trainer_config.threshold_file_path, 'w') as f:
                f.write(str(best_threshold))

            # --- Save config for Flask app ---
            config = {
                'model_name'      : 'XGB',
                'threshold'       : float(best_threshold),
                'cv_auc'          : float(cv_auc),
                'test_auc'        : float(tuned['auc']),
                'test_recall'     : float(tuned['recall']),
                'test_precision'  : float(tuned['precision']),
                'test_f2'         : float(tuned['f2']),
                'best_params'     : best_params,
                'positive_meaning': 'High Risk — Patient May Die',
                'negative_meaning': 'Low Risk — Patient May Survive',
            }
            with open(
                self.model_trainer_config.config_file_path, 'w') as f:
                json.dump(config, f, indent=4)

            logging.info(
                f"Model saved     → "
                f"{self.model_trainer_config.trained_model_file_path}"
            )
            logging.info(
                f"Threshold saved → "
                f"{self.model_trainer_config.threshold_file_path}"
            )
            logging.info(
                f"Config saved    → "
                f"{self.model_trainer_config.config_file_path}"
            )

            return tuned['auc']

        except Exception as e:
            raise CustomException(e, sys)