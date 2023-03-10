import os, sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

#A data class comes with basic functionality already implemented
@dataclass
class DataIngestionConfig:
    '''Class for initializing the Data Ingestion Configuration'''
    train_data_path: str = os.path.join('artifacts',"train.csv")
    test_data_path: str = os.path.join('artifacts',"test.csv")
    raw_data_path: str = os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion method started")
        try:
            df=pd.read_csv('notebooks\data\loan_sanction_train.csv')
            logging.info("Read the dataset as a pandas DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            train_set, test_set = train_test_split(df, test_size=0.1, random_state=33)
            logging.info("train_test_split performed")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of Data is complete")

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":
    di=DataIngestion()
    di.initiate_data_ingestion()