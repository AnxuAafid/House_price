import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logz import logging
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle 

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e :
        raise CustomException(e,sys)
    

def eval_model(X_train,Y_train,x_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            #train model
            model=list(models.values())[i]
            model.fit(X_train,Y_train)

            #predict training data

            y_train_pred = model.predict(X_train)

            # predict test data

            y_test_pred = model.predict(x_test)

            #get R2 score for train and test data

            train_model_score = r2_score(Y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]]=test_model_score

        return report

    except Exception as e:
        logging.info("Error occured while training the model")
        raise CustomException(e,sys)


def load_object(file_):
    try:
        with open(file_, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info("Error occured while loading pickle file")
        raise CustomException(e,sys)  