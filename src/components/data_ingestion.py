import os 
import sys 
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging

class DataIngestion:
    def __init__(self):
        self.raw_data_path = "artifacts/raw.csv"
        self.db_path = "artifacts/readmission_data.db"
        self.train_data_path = "artifacts/train.csv"
        self.test_data_path = "artifacts/test.csv"

    def initiate_data_ingestion(self):
        logging.info("Entered data ingetion method")

        try:
            df = pd.read_csv(
                "notebooks/data/diabetic_data.csv",
                na_values=["?"],
                keep_default_na=False
            )
            logging.info(f"Read raw data with shape {df.shape}")

            expired_hospice_ids = [11, 13, 14, 19, 20]
            before_filter = df.shape[0]
            df = df[~df["discharge_disposition_id"].isin(expired_hospice_ids)]
            logging.info(f"Filtered expired/hospice encounters: {before_filter - df.shape[0]} rows removed")

            df["readmitted_binary"] = (df["readmitted"] == "<30").astype(int)

            os.makedirs("artifacts", exist_ok=True)
            df.to_csv(self.raw_data_path, index=False)

            conn = sqlite3.connect(self.db_path)
            df.to_sql("encounters", conn, if_exists="replace", index=False)
            logging.info("Loaded raw data into SQLite")

            med_cols = ["metformin", "repaglinide", "nateglinide", "chlorpropamide",
                        "glimepiride", "glipizide", "glyburide", "pioglitazone",
                        "rosiglitazone", "acarbose", "insulin"]

            med_change_case = " + ".join(
                [f"CASE WHEN {col} IN ('Up','Down') THEN 1 ELSE 0 END" for col in med_cols]
            )

            query = f"""
            SELECT *,
                (number_outpatient + number_emergency + number_inpatient) AS Prior_Visit_Intensity,
                ({med_change_case}) AS Med_Change_Count
            FROM encounters
            """

            df_engineered = pd.read_sql(query, conn)
            conn.close()
            logging.info("SQL feature engineering complete")

            train_set, test_set = train_test_split(
                df_engineered, test_size=0.2, random_state=42,
                stratify=df_engineered["readmitted_binary"]
            )

            train_set.to_csv(self.train_data_path, index=False)
            test_set.to_csv(self.test_data_path, index=False)
            logging.info("Train/test split saved")

            return self.train_data_path, self.test_data_path

        except Exception as e:
            logging.error("Data Ingestion Failed")
            raise CustomException(e, sys)