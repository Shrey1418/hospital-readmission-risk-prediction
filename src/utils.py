import os
import sys
import math
import dill
import pandas as pd
from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def categorize_icd9(code):
    if pd.isna(code):
        return "unknown"
    code = str(code)
    if code.startswith(("V", "E")):
        return "other"
    try:
        c = float(code)
    except ValueError:
        return "unknown"
    if not math.isfinite(c):
        return "unknown"
    base = int(c)
    if 390 <= c < 460 or base == 785:
        return "circulatory"
    if 460 <= c < 520 or base == 786:
        return "respiratory"
    if 520 <= c < 580 or base == 787:
        return "digestive"
    if base == 250:
        return "diabetes"
    if 800 <= c < 1000:
        return "injury"
    if 710 <= c < 740:
        return "musculoskeletal"
    if 580 <= c < 630 or base == 788:
        return "genitourinary"
    if 140 <= c < 240:
        return "neoplasms"
    return "other"