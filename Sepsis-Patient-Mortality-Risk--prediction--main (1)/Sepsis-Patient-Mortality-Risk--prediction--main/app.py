# ================================================================================
# app.py — ICU Sepsis Mortality Prediction — Flask Web App
# ================================================================================

import os
import sys
import json

# --- Fix Python path ---
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from flask import Flask, request, render_template, jsonify

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger                     import logging
from src.exception                  import CustomException

# ================================================================================
# FLASK APP INIT
# ================================================================================

application = Flask(__name__)
app         = application

# ================================================================================
# LOAD MODEL CONFIG (for displaying metrics in UI)
# ================================================================================

def load_model_config() -> dict:
    config_path = os.path.join('artifacts', 'model_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {
        'model_name'     : 'XGB',
        'threshold'      : 0.35,
        'test_auc'       : 0.8354,
        'test_recall'    : 0.5119,
        'test_precision' : 0.6303,
        'test_f2'        : 0.5319,
    }

MODEL_CONFIG = load_model_config()

# ================================================================================
# ROUTES
# ================================================================================

# --- Home page ---
@app.route('/')
def index():
    return render_template('index.html', config=MODEL_CONFIG)


# --- Prediction page ---
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == 'GET':
        return render_template('home.html', config=MODEL_CONFIG)

    # --- POST : process form submission ---
    try:
        logging.info("Form submission received")

        data = CustomData(
            # Demographics
            max_age              = float(request.form.get('max_age', 65)),
            gender_M             = int(request.form.get('gender_M', 0)),
            los_icu              = float(request.form.get('los_icu', 5)),

            # Severity
            sofa_score           = float(request.form.get('sofa_score', 8)),

            # Vital signs
            heart_rate_mean      = float(request.form.get('heart_rate_mean', 90)),
            sbp_mean             = float(request.form.get('sbp_mean', 112)),
            dbp_mean             = float(request.form.get('dbp_mean', 61)),
            resp_rate_mean       = float(request.form.get('resp_rate_mean', 20)),
            spo2_mean            = float(request.form.get('spo2_mean', 96)),
            spo2_max             = float(request.form.get('spo2_max', 99)),
            temperature_avg      = float(request.form.get('temperature_avg', 37)),

            # Lab results
            glucose_average      = float(request.form.get('glucose_average', 150)),
            sodium_average       = float(request.form.get('sodium_average', 138)),
            avg_urineoutput      = float(request.form.get('avg_urineoutput', 1500)),

            # Comorbidities
            diabetes_without_cc  = int(request.form.get('diabetes_without_cc', 0)),
            diabetes_with_cc     = int(request.form.get('diabetes_with_cc', 0)),
            renal_disease        = int(request.form.get('renal_disease', 0)),
            severe_liver_disease = int(request.form.get('severe_liver_disease', 0)),
            aids                 = int(request.form.get('aids', 0)),
            coma                 = int(request.form.get('coma', 0)),

            # Antibiotics
            antibiotic_Glycopeptide   = int(request.form.get(
                'antibiotic_Glycopeptide', 0)),
            antibiotic_Penicillin     = int(request.form.get(
                'antibiotic_Penicillin', 0)),
            antibiotic_Carbapenem     = int(request.form.get(
                'antibiotic_Carbapenem', 0)),
            antibiotic_Aminoglycoside = int(request.form.get(
                'antibiotic_Aminoglycoside', 0)),
            antibiotic_Oxazolidinone  = int(request.form.get(
                'antibiotic_Oxazolidinone', 0)),
            antibiotic_Sulfonamide    = int(request.form.get(
                'antibiotic_Sulfonamide', 0)),
            antibiotic_Tetracycline   = int(request.form.get(
                'antibiotic_Tetracycline', 0)),
        )

        # --- Convert to DataFrame ---
        pred_df = data.get_data_as_dataframe()
        logging.info(f"Input DataFrame shape : {pred_df.shape}")

        # --- Run prediction ---
        predict_pipeline = PredictPipeline()
        result           = predict_pipeline.predict(pred_df)

        logging.info(
            f"Prediction complete | "
            f"Outcome: {result['outcome']} | "
            f"Probability: {result['probability']}% | "
            f"Risk: {result['risk_level']}"
        )

        return render_template(
            'home.html',
            result = result,
            config = MODEL_CONFIG
        )

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        error_msg = str(CustomException(e, sys))
        return render_template(
            'home.html',
            error  = error_msg,
            config = MODEL_CONFIG
        )


# ================================================================================
# API ENDPOINT — JSON (for programmatic access / testing)
# ================================================================================

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    REST API endpoint.
    Accepts JSON, returns prediction result as JSON.

    Example request:
    {
        "max_age": 65,
        "gender_M": 1,
        "sofa_score": 9,
        ...
    }
    """
    try:
        payload = request.get_json()

        if not payload:
            return jsonify({'error': 'No JSON payload received'}), 400

        data = CustomData(
            max_age              = float(payload.get('max_age', 65)),
            gender_M             = int(payload.get('gender_M', 0)),
            los_icu              = float(payload.get('los_icu', 5)),
            sofa_score           = float(payload.get('sofa_score', 8)),
            heart_rate_mean      = float(payload.get('heart_rate_mean', 90)),
            sbp_mean             = float(payload.get('sbp_mean', 112)),
            dbp_mean             = float(payload.get('dbp_mean', 61)),
            resp_rate_mean       = float(payload.get('resp_rate_mean', 20)),
            spo2_mean            = float(payload.get('spo2_mean', 96)),
            spo2_max             = float(payload.get('spo2_max', 99)),
            temperature_avg      = float(payload.get('temperature_avg', 37)),
            glucose_average      = float(payload.get('glucose_average', 150)),
            sodium_average       = float(payload.get('sodium_average', 138)),
            avg_urineoutput      = float(payload.get('avg_urineoutput', 1500)),
            diabetes_without_cc  = int(payload.get('diabetes_without_cc', 0)),
            diabetes_with_cc     = int(payload.get('diabetes_with_cc', 0)),
            renal_disease        = int(payload.get('renal_disease', 0)),
            severe_liver_disease = int(payload.get('severe_liver_disease', 0)),
            aids                 = int(payload.get('aids', 0)),
            coma                 = int(payload.get('coma', 0)),
            antibiotic_Glycopeptide   = int(payload.get(
                'antibiotic_Glycopeptide', 0)),
            antibiotic_Penicillin     = int(payload.get(
                'antibiotic_Penicillin', 0)),
            antibiotic_Carbapenem     = int(payload.get(
                'antibiotic_Carbapenem', 0)),
            antibiotic_Aminoglycoside = int(payload.get(
                'antibiotic_Aminoglycoside', 0)),
            antibiotic_Oxazolidinone  = int(payload.get(
                'antibiotic_Oxazolidinone', 0)),
            antibiotic_Sulfonamide    = int(payload.get(
                'antibiotic_Sulfonamide', 0)),
            antibiotic_Tetracycline   = int(payload.get(
                'antibiotic_Tetracycline', 0)),
        )

        pred_df = data.get_data_as_dataframe()
        result  = PredictPipeline().predict(pred_df)

        return jsonify({
            'status'      : 'success',
            'prediction'  : result['prediction'],
            'probability' : result['probability'],
            'risk_level'  : result['risk_level'],
            'outcome'     : result['outcome'],
            'threshold'   : result['threshold'],
        })

    except Exception as e:
        logging.error(f"API prediction error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ================================================================================
# RETRAIN ENDPOINT — triggers full training pipeline
# ================================================================================

@app.route('/retrain', methods=['GET'])
def retrain():
    """
    Triggers full retraining pipeline.
    Access via: http://localhost:5000/retrain
    """
    try:
        logging.info("Retraining triggered via /retrain endpoint")

        from src.pipeline.train_pipeline import TrainPipeline
        pipeline  = TrainPipeline()
        auc_score = pipeline.run()

        # Reload config after retraining
        global MODEL_CONFIG
        MODEL_CONFIG = load_model_config()

        return jsonify({
            'status'   : 'success',
            'message'  : 'Model retrained successfully',
            'auc_score': round(auc_score, 4)
        })

    except Exception as e:
        logging.error(f"Retraining error: {str(e)}")
        return jsonify({
            'status' : 'error',
            'message': str(e)
        }), 500


# ================================================================================
# HEALTH CHECK
# ================================================================================

@app.route('/health', methods=['GET'])
def health():
    """Check if app and artifacts are ready."""
    artifacts = [
        'artifacts/model.pkl',
        'artifacts/preprocessor.pkl',
        'artifacts/threshold.txt',
        'artifacts/model_config.json',
    ]
    status = {
        f: os.path.exists(f) for f in artifacts
    }
    all_ok = all(status.values())

    return jsonify({
        'status'    : 'healthy' if all_ok else 'degraded',
        'artifacts' : status,
        'model'     : MODEL_CONFIG.get('model_name', 'unknown'),
        'threshold' : MODEL_CONFIG.get('threshold', 'unknown'),
        'test_auc'  : MODEL_CONFIG.get('test_auc',  'unknown'),
    }), 200 if all_ok else 503


# ================================================================================
# MAIN
# ================================================================================

if __name__ == "__main__":
    app.run(
        host  = '0.0.0.0',
        port  = 5000,
        debug = True
    )
# ```

# ---

# ## Final project structure
# ```
# E:\Major Project\Sepsis-Patient-Mortality-Risk-prediction\
# │
# ├── app.py                          ← Flask entry point
# ├── requirements.txt
# │
# ├── artifacts/                      ← auto-generated by pipeline
# │   ├── model.pkl
# │   ├── preprocessor.pkl
# │   ├── threshold.txt
# │   ├── model_config.json
# │   ├── raw_data.csv
# │   ├── train.csv
# │   └── test.csv
# │
# ├── src/
# │   ├── __init__.py
# │   ├── exception.py
# │   ├── logger.py
# │   ├── utils.py
# │   ├── components/
# │   │   ├── __init__.py
# │   │   ├── data_ingestion.py
# │   │   ├── data_transformation.py
# │   │   └── model_trainer.py
# │   └── pipeline/
# │       ├── __init__.py
# │       ├── train_pipeline.py
# │       └── predict_pipeline.py
# │
# └── templates/
#     ├── index.html                  ← landing page
#     └── home.html                   ← form + result page