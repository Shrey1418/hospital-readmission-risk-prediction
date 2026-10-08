import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

if __name__ == "__main__":
    try:
        train_path, test_path = DataIngestion().initiate_data_ingestion()

        train_arr, test_arr, preprocessor_path, feature_names = (
            DataTransformation().initiate_data_transformation(train_path, test_path)
        )

        X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
        X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

        model, score = ModelTrainer().initiate_model_training(train_arr, test_arr)
        print(f"Training complete. PR-AUC: {score:.4f}")

        evaluator = ModelEvaluation()
        calibrated_model = evaluator.calibrate_model(model, X_train, y_train)

        y_proba = evaluator.evaluate(calibrated_model, X_test, y_test)
        evaluator.generate_shap_summary(model, X_test, feature_names)

        best_threshold, best_savings = evaluator.cost_threshold_analysis(y_test, y_proba)
        save_object("artifacts/threshold.pkl", best_threshold)
        print(f"Optimal intervention threshold: {best_threshold}, net savings: {best_savings:,.0f}")

        evaluator.report_at_threshold(y_test, y_proba, best_threshold)

    except Exception as e:
        raise CustomException(e, sys)