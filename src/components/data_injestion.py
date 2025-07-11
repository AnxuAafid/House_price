# Data Ingestion is all about reading the dataset
import os
import sys
from src.logz import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
# Initialize the data ingestion

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifcats','train.csv')
    test_data_path:str=os.path.join('artifcats','test.csv')
    raw_data_path:str=os.path.join('artifcats','raw.csv')



#create a data ingestion class
class dataIngestion:
    def __init__(self):
        self.ingestion_config= DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            df= pd.read_csv(os.path.join("notebook/Data","gemstone.csv"))
            logging.info("Dataframe read using pandas")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("Now Train test split")
            train_csv, test_csv = train_test_split(df,test_size=.30, random_state=42)
            test_csv.to_csv(self.ingestion_config.test_data_path,index=False, header = True)
            train_csv.to_csv(self.ingestion_config.train_data_path, index=False, header = True)
            logging.info("Ingestion of data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Exeception occured")
            raise CustomException(e,sys)
        
