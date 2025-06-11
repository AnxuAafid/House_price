import sklearn as sk
from sklearn.impute import SimpleImputer # handling missing value
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.exception import CustomException
from src.logz import logging
import sys
from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifcats",'preprocessor.pkl')
class DataTransformation:
    def __init__(self):
        self.data_trensformation_config= DataTransformationConfig()
    def get_data_transformation_obj(self):
        try:
            logging.info("Transformation started")
            categorical_cols = ['cut', 'clarity', 'color']
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']
            cut_map= ["Fair",'Good', 'Very Good', 'Premium', 'Ideal']
            cut_clarity = ["I1",'SI2', 'SI1','VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
            cut_color = ['D','E','F','G','H','I','J']

            # Numerical Pipeline
            # Numerical Pipeline
            logging.info("Pipeline Initiated")
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("Scaler",StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ("encoder", OrdinalEncoder(categories=[cut_map,cut_clarity,cut_color])),
                    ('scaler',StandardScaler() )
                ]
            )

            preprocessor = ColumnTransformer([
                ('numerical_pipeline', numerical_pipeline,numerical_cols),
                ('categorical_pipeline',categorical_pipeline, categorical_cols)
            ])

            return preprocessor 
            logging.info("pipeline completed")
        except Exception as e:
            logging.info("Exception Occured in data transformation ")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path, test_path):
             try:
                 train_df = pd.read_csv(train_path)
                 test_df = pd.read_csv(test_path)
                 logging.info("Read Train and Test Data Completed")
                 logging.info(f"Training Dataframe Head : \n{train_df.head().to_string()}")
                 logging.info(f"Test Dataframe Head : \n{test_df.head().to_string()}")

                 logging.info("Obtaining preprocessing object")

                 preprocessing_obj = self.get_data_transformation_obj()


                 target_column_name = "price"
                 drop_columns = [target_column_name,'id']

                 input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
                 target_feature_train_df = train_df[target_column_name]
                 input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
                 target_feature_test_df = test_df[target_column_name]


                 logging.info("Applying preprocessing object on training and testing datasets.")
                 # transforming
                 input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
                 input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

                # converting it to array for fast access
                 train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
                 test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]


                 save_object(
                     file_path=self.data_trensformation_config.preprocessor_obj_file_path,
                     obj=preprocessing_obj
                )
                 logging.info("Pickle file saved after preprocessing")

                 return (
                      train_arr,
                      test_arr,
                      self.data_trensformation_config.preprocessor_obj_file_path
                 )
             except Exception as e:
                 logging.info("Data reaading generating error")
                 raise CustomException(e,sys) 