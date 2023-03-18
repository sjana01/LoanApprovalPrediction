import sys, os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
        


class CustomData:
    def __init__(self, Gender, Married, Dependents, Education,Self_Employed, 
                ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area):
        self.Gender = Gender
        self.Married = Married
        self.Dependents = Dependents
        self.Education = Education
        self.Self_Employed = Self_Employed
        self.ApplicantIncome = ApplicantIncome 
        self.CoapplicantIncome = CoapplicantIncome
        self.LoanAmount = LoanAmount 
        self.Loan_Amount_Term = Loan_Amount_Term 
        self.Credit_History = Credit_History 
        self.Property_Area = Property_Area

    def get_data_as_df(self):
        try:
            data_input_dict={
                "Gender": [self.Gender],
                "Married": [self.Married],
                "Dependents": [self.Dependents], 
                "Education": [self.Education], 
                "Self_Employed": [self.Self_Employed], 
                "ApplicantIncome": [self.ApplicantIncome],  
                "CoapplicantIncome": [self.CoapplicantIncome], 
                "LoanAmount": [self.LoanAmount],  
                "Loan_Amount_Term": [self.Loan_Amount_Term], 
                "Credit_History": [self.Credit_History], 
                "Property_Area": [self.Property_Area]  
            }

            return pd.DataFrame(data_input_dict)

        except Exception as e:
            raise CustomException(e,sys)


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        
        try:
            preprocessor_path = 'C:\\Users\\16044\\endtoendML\\artifacts\\preprocessor.pkl'
            model_path = 'C:\\Users\\16044\\endtoendML\\artifacts\\model.pkl'
            preprocessor = load_object(file_path=preprocessor_path)
            model = load_object(file_path=model_path)
            data_scaled = preprocessor.transform(features)
            # print(data_scaled)
            prediction = model.predict(data_scaled)
            return prediction 

        except Exception as e:
            raise CustomException(e,sys)       
        