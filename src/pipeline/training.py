
import os
import sys
from src.logz import logging
from src.exception import CustomException
from src.components.data_injestion import dataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer




#calling the class
if __name__=="__main__":
    obj = dataIngestion()
    train_data_path, test_data_path= obj.initiate_data_ingestion()
    obj_trans = DataTransformation()
    train_arr,test_arr,_ = obj_trans.initiate_data_transformation(train_data_path,test_data_path)
    model_trainer = ModelTrainer()
    model_trainer.initate_model_training(train_arr,test_arr)