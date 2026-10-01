import sys
import shap
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

class ModelEvaluation:
    def calibrate_model(self, model, X_train, y_train):
        try:
            calibrated_model = CalibratedClassifierCV(model, method="isotonic", cv=3)
            calibrated_model.fit(X_train, y_train)
            save_object("artifacts/calibrated_model.pkl", calibrated_model)
            logging.info("Model calibrated using isotonic regression")
            return calibrated_model
        except Exception as e:
            raise CustomException(e, sys)

    def evaluate(self, model, X_test, y_test):
        try:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            logging.info(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
            logging.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
            logging.info(f"PR-AUC: {average_precision_score(y_test, y_proba):.4f}")
            return y_proba
        except Exception as e:
            raise CustomException(e, sys)

    def generate_shap_summary(self, model, X_test, feature_names):
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
            plt.savefig("artifacts/shap_summary.png", bbox_inches="tight")
            logging.info("SHAP summary plot saved")
        except Exception as e:
            raise CustomException(e, sys)

    def cost_threshold_analysis(self, y_test, y_proba, intervention_cost=200, penalty_avoided=2500):
        try:
            best_threshold, best_savings = 0.5, float("-inf")
            for threshold in [i / 100 for i in range(5, 91, 5)]:
                flagged = (y_proba >= threshold).astype(int)
                interventions = flagged.sum()
                true_prevented = ((flagged == 1) & (y_test == 1)).sum()
                net_savings = (true_prevented * penalty_avoided) - (interventions * intervention_cost)
                if net_savings > best_savings:
                    best_savings, best_threshold = net_savings, threshold
            logging.info(f"Optimal threshold: {best_threshold}, net savings: {best_savings}")
            return best_threshold, best_savings
        except Exception as e:
            raise CustomException(e, sys)