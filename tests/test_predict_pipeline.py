import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline

SAMPLE = {
    "race": "Caucasian", "gender": "Female", "age": "[70-80)",
    "admission_type_id": "1", "discharge_disposition_id": "1", "admission_source_id": "7",
    "time_in_hospital": 5, "num_lab_procedures": 45, "num_procedures": 1,
    "num_medications": 15, "number_outpatient": 0, "number_emergency": 1,
    "number_inpatient": 2, "number_diagnoses": 8,
    "diag_1": "428", "diag_2": "250.01", "diag_3": "401",
    "A1Cresult": "None", "max_glu_serum": "None", "change": "Ch", "diabetesMed": "Yes",
    "metformin": "No", "repaglinide": "No", "nateglinide": "No", "chlorpropamide": "No",
    "glimepiride": "No", "glipizide": "Steady", "glyburide": "No", "pioglitazone": "No",
    "rosiglitazone": "No", "acarbose": "No", "insulin": "Up",
}


def test_probability_is_valid():
    prob = PredictPipeline().predict(pd.DataFrame([SAMPLE]))[0]
    assert 0.0 <= prob <= 1.0