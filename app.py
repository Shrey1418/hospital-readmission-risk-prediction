import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

app = FastAPI()

THRESHOLD = load_object("artifacts/threshold.pkl") if os.path.exists("artifacts/threshold.pkl") else 0.30


class PatientEncounter(BaseModel):
    race: str
    gender: str
    age: str
    admission_type_id: str
    discharge_disposition_id: str
    admission_source_id: str
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int
    diag_1: str
    diag_2: str
    diag_3: str
    A1Cresult: str
    max_glu_serum: str
    change: str
    diabetesMed: str
    metformin: str
    repaglinide: str
    nateglinide: str
    chlorpropamide: str
    glimepiride: str
    glipizide: str
    glyburide: str
    pioglitazone: str
    rosiglitazone: str
    acarbose: str
    insulin: str


@app.get("/")
def root():
    return {"message": "Hospital Readmission Risk API — visit /docs to test /predict"}


@app.post("/predict")
def predict(data: PatientEncounter):
    try:
        df = pd.DataFrame([data.dict()])
        prob = PredictPipeline().predict(df)[0]
        decision = "FLAG FOR INTERVENTION" if prob >= THRESHOLD else "STANDARD DISCHARGE"
        logging.info(f"API prediction: prob={prob:.4f}, decision={decision}")
        return {"readmission_probability": float(prob), "decision": decision}
    except CustomException as e:
        logging.error(str(e))
        raise HTTPException(status_code=500, detail="Prediction failed — check server logs")