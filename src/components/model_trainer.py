import os, sys
import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
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
            
            params={
                "Logistic Regression": {},
                "Support Vector Machine": {
                    'C': [0.5, 1.0, 5, 10], 
                    # 'penalty': ['l1','l2']
                },
                "K-Neighbors Classifier": {
                    'metric': ['euclidean', 'manhattan'],
                    'n_neighbors': [2,3,5,7,10]
                },
                "Decision Tree Classifier": {
                    'criterion':['gini', 'entropy'],
                    'max_depth': [2,3,5,10],
                    # 'max_features':['sqrt','log2'],
                },
                "Gradient Boosting Classifier":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Random Forest Classifier":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "XGBoost Classifier":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoost Classifier":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                } 
            }

            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train, 
                                               X_test=X_test, y_test=y_test,
                                               models=models, params=params)
            
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