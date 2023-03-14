import os,sys, dill
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    ''' 
    Saves the object as a pickle file in the specified file path
    '''
    
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)
    


def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for model_name in list(models):
            model = models[model_name]
            # Tain Model
            model.fit(X_train, y_train)
            # Test predictions
            y_pred = model.predict(X_test)
            # Accuracy Score
            model_test_score = accuracy_score(y_test, y_pred)
            # Keep the test score in report
            report[model_name] = model_test_score

        return report

    except Exception as e:
        raise CustomException(e,sys)
        