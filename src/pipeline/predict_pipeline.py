import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object, categorize_icd9


class PredictPipeline:
    def _engineer_features(self, df):
        med_cols = ["metformin", "repaglinide", "nateglinide", "chlorpropamide",
                    "glimepiride", "glipizide", "glyburide", "pioglitazone",
                    "rosiglitazone", "acarbose", "insulin"]
        df["Med_Change_Count"] = df[med_cols].apply(
            lambda row: sum(1 for v in row if v in ("Up", "Down")), axis=1
        )
        df["Prior_Visit_Intensity"] = (
            df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
        )
        for col in ["diag_1", "diag_2", "diag_3"]:
            df[f"{col}_category"] = df[col].apply(categorize_icd9)
        return df

    def predict(self, features: pd.DataFrame):
        try:
            features = self._engineer_features(features)
            preprocessor = load_object("artifacts/preprocessor.pkl")
            model = load_object("artifacts/calibrated_model.pkl")
            data_scaled = preprocessor.transform(features)
            prob = model.predict_proba(data_scaled)[:, 1]
            logging.info(f"Prediction generated for {len(features)} record(s)")
            return prob
        except Exception as e:
            raise CustomException(e, sys)