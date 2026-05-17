# ================================================================================
# src/pipeline/train_pipeline.py — ICU Sepsis Mortality Prediction
# ================================================================================

import os
import sys
import time

# --- Fix Python path ---
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.exception                          import CustomException
from src.logger                             import logging
from src.components.data_ingestion          import DataIngestion
from src.components.data_transformation     import DataTransformation
from src.components.model_trainer           import ModelTrainer


# ================================================================================
# TRAIN PIPELINE
# ================================================================================

class TrainPipeline:
    def __init__(self):
        self.data_ingestion      = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer       = ModelTrainer()

    def run(self):
        """
        Runs the full training pipeline:
          Step 1 → Data Ingestion
          Step 2 → Data Transformation
          Step 3 → Model Training
        """
        try:
            start_time = time.time()

            print("\n" + "="*65)
            print("  ICU SEPSIS MORTALITY — TRAINING PIPELINE STARTED")
            print("="*65)

            # ----------------------------------------------------------------
            # STEP 1 : DATA INGESTION
            # ----------------------------------------------------------------
            print("\n  [1/3] Data Ingestion ...")
            step_start = time.time()

            train_path, test_path = \
                self.data_ingestion.initiate_data_ingestion()

            step_time = time.time() - step_start
            logging.info(
                f"Step 1 complete | "
                f"Train: {train_path} | "
                f"Test: {test_path} | "
                f"Time: {step_time:.1f}s"
            )
            print(f"  [1/3] Done ✓  ({step_time:.1f}s)")
            print(f"         Train → {train_path}")
            print(f"         Test  → {test_path}")

            # ----------------------------------------------------------------
            # STEP 2 : DATA TRANSFORMATION
            # ----------------------------------------------------------------
            print("\n  [2/3] Data Transformation ...")
            step_start = time.time()

            train_arr, test_arr, preprocessor_path = \
                self.data_transformation.initiate_data_transformation(
                    train_path, test_path)

            step_time = time.time() - step_start
            logging.info(
                f"Step 2 complete | "
                f"Train array: {train_arr.shape} | "
                f"Test array: {test_arr.shape} | "
                f"Preprocessor: {preprocessor_path} | "
                f"Time: {step_time:.1f}s"
            )
            print(f"  [2/3] Done ✓  ({step_time:.1f}s)")
            print(f"         Train array  → {train_arr.shape}")
            print(f"         Test  array  → {test_arr.shape}")
            print(f"         Preprocessor → {preprocessor_path}")

            # ----------------------------------------------------------------
            # STEP 3 : MODEL TRAINING
            # ----------------------------------------------------------------
            print("\n  [3/3] Model Training + Hyperparameter Tuning ...")
            step_start = time.time()

            auc_score = self.model_trainer.initiate_model_trainer(
                train_arr, test_arr)

            step_time = time.time() - step_start
            logging.info(
                f"Step 3 complete | "
                f"AUC: {auc_score:.4f} | "
                f"Time: {step_time:.1f}s"
            )
            print(f"  [3/3] Done ✓  ({step_time:.1f}s)")

            # ----------------------------------------------------------------
            # FINAL SUMMARY
            # ----------------------------------------------------------------
            total_time = time.time() - start_time

            print("\n" + "="*65)
            print("  TRAINING PIPELINE COMPLETE")
            print("="*65)
            print(f"  Final AUC Score  : {auc_score:.4f}")
            print(f"  Total Time       : {total_time:.1f}s "
                  f"({total_time/60:.1f} min)")
            print(f"\n  Artifacts saved:")
            artifacts = [
                ('raw_data.csv',       'Full deduplicated dataset'),
                ('train.csv',          '80% train split'),
                ('test.csv',           '20% test split'),
                ('preprocessor.pkl',   'Fitted scaler + imputer'),
                ('model.pkl',          'Tuned XGBClassifier'),
                ('threshold.txt',      'F2-optimal threshold'),
                ('model_config.json',  'Model config for Flask app'),
            ]
            for fname, desc in artifacts:
                fpath  = os.path.join('artifacts', fname)
                exists = os.path.exists(fpath)
                size   = (os.path.getsize(fpath) / 1024
                          if exists else 0)
                status = f"✓  ({size:.1f} KB)" if exists else "✗  MISSING"
                print(f"    {status:<18} artifacts/{fname:<25} {desc}")
            print("="*65)

            logging.info(
                f"Training pipeline complete | "
                f"AUC: {auc_score:.4f} | "
                f"Total time: {total_time:.1f}s"
            )

            return auc_score

        except Exception as e:
            logging.error(f"Training pipeline failed: {str(e)}")
            raise CustomException(e, sys)


# ================================================================================
# MAIN
# ================================================================================

if __name__ == "__main__":
    pipeline  = TrainPipeline()
    auc_score = pipeline.run()