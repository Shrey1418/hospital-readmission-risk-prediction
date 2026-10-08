import sys
import shap
import numpy as np
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
            plt.close()

            sv = shap_values[1] if isinstance(shap_values, list) else shap_values
            mean_abs = np.abs(sv).mean(axis=0)
            top = sorted(zip(feature_names, mean_abs), key=lambda x: -x[1])[:10]
            print("\nTop 10 features by mean |SHAP|:")
            for name, val in top:
                line = f"{name}: {val:.4f}"
                print(line)
                logging.info(f"SHAP top feature - {line}")
            logging.info("SHAP summary plot saved")
        except Exception as e:
            raise CustomException(e, sys)

    def report_at_threshold(self, y_test, y_proba, threshold):
        try:
            y_test = np.asarray(y_test)
            flagged = (y_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, flagged).ravel()
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            msg = (f"At threshold {threshold}: flagged={flagged.sum()} ({flagged.mean():.1%} of patients), "
                   f"TP={tp}, FP={fp}, FN={fn}, TN={tn}, precision={precision:.3f}, recall={recall:.3f}")
            print(msg)
            logging.info(msg)
        except Exception as e:
            raise CustomException(e, sys)

    def cost_threshold_analysis(self, y_test, y_proba, intervention_cost=200, penalty_avoided=2500, success_rate=0.3):
        try:
            y_test = np.asarray(y_test)
            thresholds = [i / 100 for i in range(5, 96, 5)]

            def net_savings(flagged, rate):
                prevented = ((flagged == 1) & (y_test == 1)).sum() * rate
                return prevented * penalty_avoided - flagged.sum() * intervention_cost

            print("\nSensitivity: best policy vs intervention success rate")
            for rate in [0.1, 0.2, 0.3, 0.5, 1.0]:
                results = {t: net_savings((y_proba >= t).astype(int), rate) for t in thresholds}
                t_best = max(results, key=results.get)
                everyone = net_savings(np.ones(len(y_test), dtype=int), rate)
                line = (f"success_rate={rate}: best threshold={t_best}, "
                        f"net savings={results[t_best]:,.0f}, "
                        f"call-everyone={everyone:,.0f}, call-no-one=0")
                print(line)
                logging.info(line)

            results = {t: net_savings((y_proba >= t).astype(int), success_rate) for t in thresholds}
            best_threshold = max(results, key=results.get)
            best_savings = results[best_threshold]
            if best_savings <= 0:
                logging.warning("No profitable intervention policy at this success rate")
            logging.info(f"Headline (success_rate={success_rate}): threshold={best_threshold}, net savings={best_savings:.0f}")
            return best_threshold, best_savings
        except Exception as e:
            raise CustomException(e, sys)