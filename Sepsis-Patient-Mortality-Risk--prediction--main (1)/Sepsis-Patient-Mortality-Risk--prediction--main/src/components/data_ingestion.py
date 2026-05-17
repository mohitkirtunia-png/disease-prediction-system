import os
import sys

# --- Fix Python path ---
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger    import logging

from src.components.data_transformation import DataTransformation
from src.components.model_trainer       import ModelTrainer


@dataclass
class DataIngestionConfig:
    raw_data_path  : str = os.path.join('artifacts', 'raw_data.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path : str = os.path.join('artifacts', 'test.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            source_path = (
                r'E:\Major Project\Sepsis-Patient-Mortality-Risk-prediction'
                r'\Experiments\experiment_1\Dataset\Sepsis_data_before_cleaning.csv'
            )
            df = pd.read_csv(source_path)
            logging.info(f"Raw dataset loaded | Shape: {df.shape}")

            # --- Remove duplicates on FULL dataset BEFORE split ---
            before = len(df)
            df     = df.drop_duplicates().reset_index(drop=True)
            logging.info(
                f"Duplicates removed : {before - len(df)} rows | "
                f"Remaining: {len(df)}"
            )

            # --- Split by subject_id to prevent patient leakage ---
            if 'subject_id' in df.columns:
                logging.info("Splitting by subject_id to prevent leakage")

                subject_labels = df.groupby('subject_id')[
                    'hospital_expire_flag'].max().reset_index()

                train_ids, test_ids = train_test_split(
                    subject_labels['subject_id'],
                    test_size    = 0.2,
                    random_state = 42,
                    stratify     = subject_labels['hospital_expire_flag']
                )

                train_set = df[df['subject_id'].isin(train_ids)].copy()
                test_set  = df[df['subject_id'].isin(test_ids)].copy()

                overlap = set(train_set['subject_id']) & \
                          set(test_set['subject_id'])
                logging.info(
                    f"Subject ID overlap after fix : {len(overlap)} rows")
                assert len(overlap) == 0, \
                    "ERROR: subject_id overlap detected!"

            else:
                logging.warning(
                    "subject_id not found — using standard stratified split")
                train_set, test_set = train_test_split(
                    df,
                    test_size    = 0.2,
                    random_state = 42,
                    stratify     = df['hospital_expire_flag']
                )

            # --- Save ---
            os.makedirs(
                os.path.dirname(self.ingestion_config.raw_data_path),
                exist_ok=True
            )
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False, header=True)
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False, header=True)
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False, header=True)

            logging.info(f"Train size           : {train_set.shape}")
            logging.info(f"Test  size           : {test_set.shape}")
            logging.info(
                f"Train mortality rate : "
                f"{train_set['hospital_expire_flag'].mean()*100:.2f}%"
            )
            logging.info(
                f"Test  mortality rate : "
                f"{test_set['hospital_expire_flag'].mean()*100:.2f}%"
            )
            logging.info("Data Ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj                    = DataIngestion()
    train_path, test_path  = obj.initiate_data_ingestion()

    data_transformation        = DataTransformation()
    train_arr, test_arr, _     = \
        data_transformation.initiate_data_transformation(
            train_path, test_path)

    model_trainer  = ModelTrainer()
    auc_score      = model_trainer.initiate_model_trainer(
                         train_arr, test_arr)

    print(f"\n  Pipeline Complete | Final AUC : {auc_score:.4f}")