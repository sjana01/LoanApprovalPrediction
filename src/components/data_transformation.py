import os, sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        '''
        This function returns the preprocessor with the numerical and categorical pipeline
        '''

        try:
            numerical_columns = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
            categorical_columns = ["Gender","Married", "Dependents", "Education", "Self_Employed", "Property_Area"]

            numerical_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
                ]
            )
            logging.info("Numerical pipeline completed")

            categorical_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(sparse=False)),
                ("scaler", StandardScaler())
                ]
            )
            logging.info("Categorical pipeline completed")

            preprocessor = ColumnTransformer([
                ("numerical_pipelines", numerical_pipeline, numerical_columns),
                ("categorical_pipelines", categorical_pipeline, categorical_columns)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        ''' 
        This function returns the transformed data ready for model training
        '''

        try:
            # read data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and Test data read successfully")
            # initiate preprocessor object
            preprocessor_obj = self.get_data_transformer_object()
            
            # set the target column
            target_column = "Loan_Status"

            X_train_df = train_df.drop(columns=[target_column, "Loan_ID"], axis=1)
            y_train_df = train_df[target_column]

            X_test_df = test_df.drop(columns=[target_column, "Loan_ID"], axis=1)
            y_test_df = test_df[target_column]

            # fit-transform X
            X_train_arr = preprocessor_obj.fit_transform(X_train_df)
            X_test_arr = preprocessor_obj.fit_transform(X_test_df)

            # make y an numpy array of 0 and 1
            y_train_arr = np.array((y_train_df == 'Y').astype(int))
            y_test_arr = np.array((y_test_df == 'Y').astype(int))

            logging.info("Applied preprocessing object on train and test data")

            train_arr = np.column_stack([X_train_arr, y_train_arr])
            test_arr = np.column_stack([X_test_arr, y_test_arr])

            # save the preprocessor object as a pickle file 
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
                )
            
            logging.info("Saved preprocessor object")

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e,sys)

