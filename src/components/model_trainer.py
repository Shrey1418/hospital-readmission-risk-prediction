import sys
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import average_precision_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

class ModelTrainer:
    def __init__(self):
        self.model_path = "artifacts/model.pkl"

    def initiate_model_training(self, train_array, test_array):
        try:
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            logging.info("Split arrays into features and target")

            baseline = LogisticRegression(max_iter=1000, class_weight="balanced")
            baseline.fit(X_train, y_train)
            baseline_score = average_precision_score(y_test, baseline.predict_proba(X_test)[:, 1])
            logging.info(f"Baseline PR-AUC: {baseline_score:.4f}")

            param_dist = {
                "max_depth": [3, 4, 5, 6],
                "learning_rate": [0.01, 0.05, 0.1],
                "n_estimators": [100, 200, 300],
                "scale_pos_weight": [3, 5, 8]
            }
            xgb = XGBClassifier(eval_metric="logloss", random_state=42)
            search = RandomizedSearchCV(xgb, param_dist, scoring="average_precision", cv=3, n_iter=15, random_state=42)
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            xgb_score = average_precision_score(y_test, best_model.predict_proba(X_test)[:, 1])
            logging.info(f"Tuned XGBoost PR-AUC: {xgb_score:.4f}, best params: {search.best_params_}")

            final_model = best_model if xgb_score > baseline_score else baseline
            save_object(self.model_path, final_model)

            return final_model, xgb_score

        except Exception as e:
            logging.error("Model training failed")
            raise CustomException(e, sys)