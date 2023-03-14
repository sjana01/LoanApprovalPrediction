import os, sys
import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test input feature and taget feature")
            X_train, y_train, X_test, y_test = (train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1])

            models = {"Logistic Regression" : LogisticRegression(),
                      "Support Vector Machine" : LinearSVC(),
                      "K-Neighbors Classifier" : KNeighborsClassifier(),
                      "Decision Tree Classifier" : DecisionTreeClassifier(),
                      "Gradient Boosting Classifier" : GradientBoostingClassifier(),
                      "Random Forest Classifier" : RandomForestClassifier(),
                      "XGBoost Classifier" : XGBClassifier(),
                      "CatBoost Classifier" : CatBoostClassifier(verbose=False)
                      }
            
            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train, 
                                               X_test=X_test, y_test=y_test,
                                               models=models)
            
            # Get the best test score from the dict
            best_model_score = max(sorted(model_report.values()))
            # Get the best model
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            logging.info("Best model found on test data")

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            predicted = best_model.predict(X_test)
            score = accuracy_score(y_test, predicted)

            return score


        except Exception as e:
            raise CustomException(e,sys)



if __name__=="__main__":
    data_ingestion = DataIngestion()
    train_path,test_path = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_array, test_array, _ = data_transformation.initiate_data_transformation(train_path,test_path)

    model_trainer = ModelTrainer()
    model_score = model_trainer.initiate_model_trainer(train_array, test_array)
    print(model_score)