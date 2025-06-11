import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logz import logging
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from src.utils import save_object, eval_model

@dataclass

class ModelTrainConfig:
    train_model_file_path = os.path.join("artifcats","model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainConfig()
    def initate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting Dependent and Independent varaibles from train and test")
            xtrain,ytrain,xtest,ytest=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )


            #Train multiple models
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "ElasticNet Regression": ElasticNet()
            }

            model_report:dict=eval_model(xtrain,ytrain,xtest,ytest,models)

            print(model_report)
            print("+"*30)
            logging.info(f"Model_report: {model_report}")

            # to get best model

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]

            print(f"Best model is {best_model_name}, with R2_score {best_model_score}")
            print("="*30)
            logging.info(f"Best model is {best_model_name}, with R2_score {best_model_score}")
            #logging.infor("Hyperparameter tuning started for catboost ")

            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.info("Error occured in model training")
            raise CustomException(e,sys)