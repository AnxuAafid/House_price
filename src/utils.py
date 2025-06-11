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