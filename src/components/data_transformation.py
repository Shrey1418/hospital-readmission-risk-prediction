import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

class DataTransformation:
    def __init__(self):
        self.preprocessor_obj_path = "artifacts/preprocessor.pkl"

    def drop_unusable_columns(self, df):
        try:
            cols_to_drop = ["weight", "payer_code", "medical_specialty",
                             "encounter_id", "patient_nbr", "readmitted"]
            df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
            logging.info(f"Dropped unusable columns: {cols_to_drop}")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def bucket_diagnosis_codes(self, df):
        try:
            def categorize_icd9(code):
                try:
                    code = str(code)
                    if code.startswith(("V", "E")):
                        return "other"
                    code_num = float(code)
                    if 390 <= code_num <= 459 or code_num == 785:
                        return "circulatory"
                    elif 460 <= code_num <= 519 or code_num == 786:
                        return "respiratory"
                    elif 520 <= code_num <= 579 or code_num == 787:
                        return "digestive"
                    elif code_num == 250:
                        return "diabetes"
                    elif 800 <= code_num <= 999:
                        return "injury"
                    elif 710 <= code_num <= 739:
                        return "musculoskeletal"
                    elif 580 <= code_num <= 629 or code_num == 788:
                        return "genitourinary"
                    elif 140 <= code_num <= 239:
                        return "neoplasms"
                    else:
                        return "other"
                except (ValueError, TypeError):
                    return "unknown"

            for col in ["diag_1", "diag_2", "diag_3"]:
                df[f"{col}_category"] = df[col].apply(categorize_icd9)
            df = df.drop(columns=["diag_1", "diag_2", "diag_3"])
            logging.info("Bucketed ICD9 diagnosis codes into categories")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def clean_data(self, df):
        try:
            df["race"] = df["race"].fillna("Unknown").astype(str)
            df["gender"] = df["gender"].astype(str)
            df["age"] = df["age"].astype(str)
            df["A1Cresult"] = df["A1Cresult"].astype(str)
            df["max_glu_serum"] = df["max_glu_serum"].astype(str)
            df["admission_type_id"] = df["admission_type_id"].astype(str)
            df["discharge_disposition_id"] = df["discharge_disposition_id"].astype(str)
            df["admission_source_id"] = df["admission_source_id"].astype(str)
            logging.info("Cleaned and type-cast categorical columns")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info(f"Read train {train_df.shape}, test {test_df.shape}")

            train_df = self.drop_unusable_columns(train_df)
            test_df = self.drop_unusable_columns(test_df)

            train_df = self.bucket_diagnosis_codes(train_df)
            test_df = self.bucket_diagnosis_codes(test_df)

            train_df = self.clean_data(train_df)
            test_df = self.clean_data(test_df)

            target_column = "readmitted_binary"

            numeric_features = ["time_in_hospital", "num_lab_procedures", "num_procedures",
                                 "num_medications", "number_outpatient", "number_emergency",
                                 "number_inpatient", "number_diagnoses",
                                 "Prior_Visit_Intensity", "Med_Change_Count"]
            categorical_features = ["race", "gender", "age", "diag_1_category",
                                     "diag_2_category", "diag_3_category",
                                     "change", "diabetesMed",
                                     "admission_type_id", "discharge_disposition_id",
                                     "admission_source_id", "A1Cresult", "max_glu_serum"]

            preprocessor = self.get_preprocessor_object(numeric_features, categorical_features)

            X_train = train_df[numeric_features + categorical_features]
            y_train = train_df[target_column]
            X_test = test_df[numeric_features + categorical_features]
            y_test = test_df[target_column]

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            feature_names = preprocessor.get_feature_names_out()
            logging.info(f"Preprocessing complete, {len(feature_names)} output columns")

            save_object(self.preprocessor_obj_path, preprocessor)

            train_arr = np.c_[X_train_transformed, np.array(y_train)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            return train_arr, test_arr, self.preprocessor_obj_path, feature_names

        except Exception as e:
            logging.error("Data transformation failed")
            raise CustomException(e, sys)