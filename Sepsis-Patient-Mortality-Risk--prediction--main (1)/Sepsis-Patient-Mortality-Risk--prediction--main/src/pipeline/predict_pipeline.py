# ================================================================================
# src/pipeline/predict_pipeline.py — ICU Sepsis Mortality Prediction
# ================================================================================

import os
import sys
import json
import numpy as np
import pandas as pd

# --- Fix Python path ---
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.exception import CustomException
from src.logger    import logging
from src.utils     import load_object


# ================================================================================
# PREDICT PIPELINE
# ================================================================================

class PredictPipeline:
    def __init__(self):
        self.model_path        = os.path.join('artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        self.threshold_path    = os.path.join('artifacts', 'threshold.txt')
        self.config_path       = os.path.join('artifacts', 'model_config.json')

    def load_threshold(self) -> float:
        """Load optimal threshold saved during training."""
        try:
            if os.path.exists(self.threshold_path):
                with open(self.threshold_path, 'r') as f:
                    threshold = float(f.read().strip())
                logging.info(f"Threshold loaded : {threshold}")
                return threshold
            else:
                logging.warning(
                    "threshold.txt not found — using default 0.35")
                return 0.35
        except Exception as e:
            raise CustomException(e, sys)

    def load_config(self) -> dict:
        """Load model config saved during training."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame) -> dict:
        """
        Takes a DataFrame of patient features,
        returns prediction dict with probability,
        risk level and outcome label.
        """
        try:
            logging.info("PredictPipeline.predict() called")

            # --- Load artifacts ---
            model        = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)
            threshold    = self.load_threshold()

            logging.info(
                f"Artifacts loaded | "
                f"Model: {self.model_path} | "
                f"Threshold: {threshold}"
            )

            # --- Scale features ---
            data_scaled  = preprocessor.transform(features)

            # --- Predict probability ---
            probability  = model.predict_proba(data_scaled)[0][1]
            prediction   = int(probability >= threshold)

            # --- Risk level ---
            if probability >= 0.70:
                risk_level = 'Critical'
                risk_color = '#dc3545'
            elif probability >= threshold:
                risk_level = 'High'
                risk_color = '#fd7e14'
            elif probability >= 0.25:
                risk_level = 'Moderate'
                risk_color = '#ffc107'
            else:
                risk_level = 'Low'
                risk_color = '#28a745'

            outcome = (
                'High Risk — Patient May Die'
                if prediction == 1
                else 'Low Risk — Patient May Survive'
            )

            result = {
                'prediction'  : prediction,
                'probability' : round(float(probability) * 100, 2),
                'risk_level'  : risk_level,
                'risk_color'  : risk_color,
                'outcome'     : outcome,
                'threshold'   : round(threshold * 100, 1),
            }

            logging.info(
                f"Prediction result | "
                f"Prob: {result['probability']}% | "
                f"Risk: {risk_level} | "
                f"Threshold: {threshold}"
            )

            return result

        except Exception as e:
            raise CustomException(e, sys)


# ================================================================================
# CUSTOM DATA — Maps form inputs to model feature DataFrame
# ================================================================================

class CustomData:
    def __init__(
        self,
        max_age              : float,
        gender_M             : int,
        los_icu              : float,
        sofa_score           : float,
        heart_rate_mean      : float,
        sbp_mean             : float,
        dbp_mean             : float,
        resp_rate_mean       : float,
        spo2_mean            : float,
        spo2_max             : float,
        temperature_avg      : float,
        glucose_average      : float,
        sodium_average       : float,
        avg_urineoutput      : float,
        diabetes_without_cc  : int,
        diabetes_with_cc     : int,
        renal_disease        : int,
        severe_liver_disease : int,
        aids                 : int,
        coma                 : int,
        antibiotic_Glycopeptide   : int,
        antibiotic_Penicillin     : int,
        antibiotic_Carbapenem     : int,
        antibiotic_Aminoglycoside : int,
        antibiotic_Oxazolidinone  : int,
        antibiotic_Sulfonamide    : int,
        antibiotic_Tetracycline   : int,
    ):
        # Demographics
        self.max_age               = max_age
        self.gender_M              = gender_M
        self.los_icu               = los_icu

        # Severity
        self.sofa_score            = sofa_score

        # Vital signs
        self.heart_rate_mean       = heart_rate_mean
        self.sbp_mean              = sbp_mean
        self.dbp_mean              = dbp_mean
        self.resp_rate_mean        = resp_rate_mean
        self.spo2_mean             = spo2_mean
        self.spo2_max              = spo2_max
        self.temperature_avg       = temperature_avg

        # Lab results
        self.glucose_average       = glucose_average
        self.sodium_average        = sodium_average
        self.avg_urineoutput       = avg_urineoutput

        # Comorbidities
        self.diabetes_without_cc   = diabetes_without_cc
        self.diabetes_with_cc      = diabetes_with_cc
        self.renal_disease         = renal_disease
        self.severe_liver_disease  = severe_liver_disease
        self.aids                  = aids
        self.coma                  = coma

        # Antibiotics
        self.antibiotic_Glycopeptide   = antibiotic_Glycopeptide
        self.antibiotic_Penicillin     = antibiotic_Penicillin
        self.antibiotic_Carbapenem     = antibiotic_Carbapenem
        self.antibiotic_Aminoglycoside = antibiotic_Aminoglycoside
        self.antibiotic_Oxazolidinone  = antibiotic_Oxazolidinone
        self.antibiotic_Sulfonamide    = antibiotic_Sulfonamide
        self.antibiotic_Tetracycline   = antibiotic_Tetracycline

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """
        Converts form input into a single-row DataFrame
        matching the exact feature names used during training.
        """
        try:
            # Load preprocessor to get exact feature name order
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            preprocessor      = load_object(preprocessor_path)

            # Get feature names from preprocessor
            expected_features = (
                preprocessor
                .named_transformers_['num_pipeline']
                .feature_names_in_
                if hasattr(
                    preprocessor
                    .named_transformers_['num_pipeline'],
                    'feature_names_in_'
                )
                else None
            )

            # Build input dict with all features
            input_dict = {
                'max_age'                    : [self.max_age],
                'gender_F'                   : [1 - self.gender_M],
                'gender_M'                   : [self.gender_M],
                'los_icu'                    : [self.los_icu],
                'sofa_score'                 : [self.sofa_score],
                'avg_urineoutput'            : [self.avg_urineoutput],
                'temperature_min'            : [self.temperature_avg - 0.5],
                'temperature_max'            : [self.temperature_avg + 0.5],
                'temperature_avg'            : [self.temperature_avg],
                'glucose_min'                : [self.glucose_average * 0.85],
                'glucose_max'                : [self.glucose_average * 1.15],
                'glucose_average'            : [self.glucose_average],
                'sodium_min'                 : [self.sodium_average - 2],
                'sodium_max'                 : [self.sodium_average + 2],
                'sodium_average'             : [self.sodium_average],
                'heart_rate_min'             : [self.heart_rate_mean - 10],
                'heart_rate_max'             : [self.heart_rate_mean + 10],
                'heart_rate_mean'            : [self.heart_rate_mean],
                'resp_rate_min'              : [self.resp_rate_mean - 3],
                'resp_rate_max'              : [self.resp_rate_mean + 3],
                'resp_rate_mean'             : [self.resp_rate_mean],
                'spo2_min'                   : [self.spo2_mean - 2],
                'spo2_max'                   : [self.spo2_max],
                'spo2_mean'                  : [self.spo2_mean],
                'sbp_min'                    : [self.sbp_mean - 15],
                'sbp_max'                    : [self.sbp_mean + 15],
                'sbp_mean'                   : [self.sbp_mean],
                'dbp_min'                    : [self.dbp_mean - 10],
                'dbp_max'                    : [self.dbp_mean + 10],
                'dbp_mean'                   : [self.dbp_mean],
                'diabetes_without_cc'        : [self.diabetes_without_cc],
                'diabetes_with_cc'           : [self.diabetes_with_cc],
                'severe_liver_disease'       : [self.severe_liver_disease],
                'aids'                       : [self.aids],
                'renal_disease'              : [self.renal_disease],
                'coma'                       : [self.coma],
                'race_Black'                 : [0],
                'race_Hispanic'              : [0],
                'race_White'                 : [0],
                'race_Other'                 : [0],
                'antibiotic_Carbapenem'      : [self.antibiotic_Carbapenem],
                'antibiotic_Aminoglycoside'  : [self.antibiotic_Aminoglycoside],
                'antibiotic_Glycopeptide'    : [self.antibiotic_Glycopeptide],
                'antibiotic_Oxazolidinone'   : [self.antibiotic_Oxazolidinone],
                'antibiotic_Penicillin'      : [self.antibiotic_Penicillin],
                'antibiotic_Sulfonamide'     : [self.antibiotic_Sulfonamide],
                'antibiotic_Tetracycline'    : [self.antibiotic_Tetracycline],
            }

            df = pd.DataFrame(input_dict)

            # --- Reorder columns to match training feature order ---
            if expected_features is not None:
                # Keep only columns the model was trained on
                # in the exact same order
                df = df.reindex(columns=expected_features, fill_value=0)
                logging.info(
                    f"Features aligned to training order: "
                    f"{len(expected_features)} cols"
                )
            else:
                logging.warning(
                    "Could not retrieve feature order from preprocessor. "
                    "Using dict order."
                )

            logging.info(f"CustomData DataFrame shape: {df.shape}")
            return df

        except Exception as e:
            raise CustomException(e, sys)


# ================================================================================
# QUICK TEST — run directly to verify pipeline works
# ================================================================================

if __name__ == "__main__":

    # --- Create a sample patient ---
    sample_patient = CustomData(
        max_age              = 65,
        gender_M             = 1,
        los_icu              = 5.0,
        sofa_score           = 9,
        heart_rate_mean      = 95.0,
        sbp_mean             = 108.0,
        dbp_mean             = 60.0,
        resp_rate_mean       = 22.0,
        spo2_mean            = 95.0,
        spo2_max             = 99.0,
        temperature_avg      = 37.2,
        glucose_average      = 160.0,
        sodium_average       = 138.0,
        avg_urineoutput      = 1200.0,
        diabetes_without_cc  = 1,
        diabetes_with_cc     = 0,
        renal_disease        = 1,
        severe_liver_disease = 0,
        aids                 = 0,
        coma                 = 0,
        antibiotic_Glycopeptide   = 1,
        antibiotic_Penicillin     = 0,
        antibiotic_Carbapenem     = 1,
        antibiotic_Aminoglycoside = 0,
        antibiotic_Oxazolidinone  = 0,
        antibiotic_Sulfonamide    = 0,
        antibiotic_Tetracycline   = 0,
    )

    # --- Convert to DataFrame ---
    df = sample_patient.get_data_as_dataframe()
    print(f"\nInput DataFrame shape : {df.shape}")
    print(f"Columns               : {df.columns.tolist()}")

    # --- Run prediction ---
    pipeline = PredictPipeline()
    result   = pipeline.predict(df)

    print("\n" + "="*50)
    print("PREDICTION RESULT")
    print("="*50)
    print(f"  Outcome      : {result['outcome']}")
    print(f"  Probability  : {result['probability']}%")
    print(f"  Risk Level   : {result['risk_level']}")
    print(f"  Threshold    : {result['threshold']}%")
    print("="*50)