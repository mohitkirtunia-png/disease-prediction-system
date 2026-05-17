import os
import sys

# --- Fix Python path ---
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import re
import numpy as np
import pandas as pd

from dataclasses            import dataclass
from imblearn.over_sampling import SMOTE

from sklearn.compose        import ColumnTransformer
from sklearn.impute         import SimpleImputer
from sklearn.pipeline       import Pipeline
from sklearn.preprocessing  import StandardScaler

from src.exception import CustomException
from src.logger    import logging
from src.utils     import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        'artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [
            re.sub(r'[^A-Za-z0-9_]', '_', col).strip('_')
            for col in df.columns
        ]
        seen     = {}
        new_cols = []
        for col in df.columns:
            if col in seen:
                seen[col] += 1
                new_cols.append(f'{col}_{seen[col]}')
            else:
                seen[col] = 0
                new_cols.append(col)
        df.columns = new_cols
        return df

    def fix_data_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if 'avg_urineoutput' in df.columns:
            neg = (df['avg_urineoutput'] < 0).sum()
            df['avg_urineoutput'] = df['avg_urineoutput'].apply(
                lambda x: np.nan if x < 0 else x)
            logging.info(f"Negative urine output fixed: {neg} → NaN")

        temp_cols = [c for c in df.columns if 'temperature' in c.lower()]
        for col in temp_cols:
            bad = (df[col] < 30).sum()
            if bad > 0:
                df[col] = df[col].apply(
                    lambda x: np.nan if x < 30 else x)
                logging.info(f"Unrealistic {col}: {bad} → NaN")

        if 'glucose_average' in df.columns:
            extreme = (df['glucose_average'] > 1500).sum()
            df['glucose_average'] = df['glucose_average'].apply(
                lambda x: np.nan if x > 1500 else x)
            logging.info(f"Extreme glucose: {extreme} → NaN")

        return df

    def get_data_transformer_object(self, numerical_columns: list):
        try:
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler',  StandardScaler())
            ])
            preprocessor = ColumnTransformer(transformers=[
                ('num_pipeline', num_pipeline, numerical_columns)
            ])
            logging.info(
                f"Preprocessor built | "
                f"Numeric cols: {len(numerical_columns)}")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self,
                                      train_path: str,
                                      test_path: str):
        try:
            logging.info("Data Transformation started")

            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            logging.info(
                f"Train: {train_df.shape} | Test: {test_df.shape}")

            # --- Check subject_id overlap ---
            if 'subject_id' in train_df.columns:
                overlap = set(train_df['subject_id']) & \
                          set(test_df['subject_id'])
                logging.info(
                    f"Train/Test subject_id overlap : {len(overlap)}")
                if len(overlap) > 0:
                    logging.warning(
                        f"WARNING: {len(overlap)} subject IDs in both sets!")

            # --- Clean column names ---
            train_df = self.clean_column_names(train_df)
            test_df  = self.clean_column_names(test_df)
            logging.info("Column names cleaned")

            # --- Fix data errors ---
            train_df = self.fix_data_errors(train_df)
            test_df  = self.fix_data_errors(test_df)

            # --- Drop subject_id ---
            if 'subject_id' in train_df.columns:
                train_df.drop(columns=['subject_id'], inplace=True)
                test_df.drop(columns=['subject_id'],  inplace=True)
                logging.info("subject_id dropped")

            # --- Remove constant features (from train only) ---
            target_col    = 'hospital_expire_flag'
            constant_cols = [
                col for col in train_df.columns
                if col != target_col
                and train_df[col].nunique() <= 1
            ]
            if constant_cols:
                train_df.drop(columns=constant_cols, inplace=True)
                test_df.drop(columns=constant_cols,  inplace=True)
                logging.info(f"Constant features removed: {constant_cols}")

            # --- Drop high missing > 40% (from train only) ---
            missing_pct = train_df.isnull().mean()
            high_miss   = missing_pct[
                (missing_pct > 0.40) &
                (missing_pct.index != target_col)
            ].index.tolist()
            if high_miss:
                train_df.drop(columns=high_miss, inplace=True)
                test_df.drop(columns=high_miss,  inplace=True)
                logging.info(f"High-missing cols dropped: {high_miss}")

            # --- Drop non-numeric columns ---
            obj_cols = [
                c for c in train_df.select_dtypes(
                    include=['object', 'category']).columns
                if c != target_col
            ]
            if obj_cols:
                train_df.drop(columns=obj_cols, inplace=True)
                test_df.drop(columns=obj_cols,  inplace=True)
                logging.info(f"Non-numeric cols dropped: {obj_cols}")

            # --- Separate features and target ---
            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]
            X_test  = test_df.drop(columns=[target_col])
            y_test  = test_df[target_col]

            logging.info(
                f"After cleaning | "
                f"X_train: {X_train.shape} | X_test: {X_test.shape}"
            )
            logging.info(
                f"Train mortality: {y_train.mean()*100:.2f}% | "
                f"Test  mortality: {y_test.mean()*100:.2f}%"
            )

            # --- IQR Outlier Capping (bounds from train only) ---
            binary_cols     = [
                c for c in X_train.columns
                if X_train[c].nunique() <= 2
            ]
            continuous_cols = [
                c for c in X_train.columns
                if c not in binary_cols
            ]
            for col in continuous_cols:
                Q1    = X_train[col].quantile(0.25)
                Q3    = X_train[col].quantile(0.75)
                IQR   = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                X_train[col] = X_train[col].clip(lower, upper)
                X_test[col]  = X_test[col].clip(lower, upper)
            logging.info(
                "IQR outlier capping applied (bounds from train only)")

            # --- Build & fit preprocessor on TRAIN only ---
            numerical_columns = X_train.columns.tolist()
            preprocessing_obj = self.get_data_transformer_object(
                numerical_columns)

            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr  = preprocessing_obj.transform(X_test)
            logging.info("Scaling applied (fit on train only)")

            # --- SMOTE on train only ---
            logging.info(
                f"Before SMOTE | "
                f"Class 0: {(y_train==0).sum()} | "
                f"Class 1: {(y_train==1).sum()}"
            )
            smote = SMOTE(random_state=42)
            X_train_arr, y_train_res = smote.fit_resample(
                X_train_arr, y_train)
            logging.info(
                f"After  SMOTE | "
                f"Class 0: {(y_train_res==0).sum()} | "
                f"Class 1: {(y_train_res==1).sum()}"
            )

            # --- Combine into arrays ---
            train_arr = np.c_[X_train_arr, np.array(y_train_res)]
            test_arr  = np.c_[X_test_arr,  np.array(y_test)]

            logging.info(f"Final train array : {train_arr.shape}")
            logging.info(f"Final test  array : {test_arr.shape}")

            assert train_arr.shape[0] > test_arr.shape[0], \
                "ERROR: Train smaller than test — check pipeline!"

            # --- Save preprocessor ---
            save_object(
                file_path = self.data_transformation_config\
                    .preprocessor_obj_file_path,
                obj       = preprocessing_obj
            )
            logging.info(
                f"Preprocessor saved → "
                f"{self.data_transformation_config.preprocessor_obj_file_path}"
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)