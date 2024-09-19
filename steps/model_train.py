import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
import mlflow
from zenml.client import Client

experiment_tracker=Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(xtrain:pd.DataFrame,
                xtest:pd.DataFrame,
                ytrain:pd.DataFrame,
                ytest:pd.DataFrame,
                config:ModelNameConfig):
    
    try:
        model=None
        if config.model_name=="LinearRegression":
            mlflow.sklearn.autolog()
            model=LinearRegressionModel()
            trained_model=model.train(xtrain,ytrain)
            return trained_model
        else:
            raise ValueError("Model {} not supported ".format(config.model_name))
    except Exception as e:
        logging.error("Error in training model:{}".format(e))
        raise e
     
    
    
  