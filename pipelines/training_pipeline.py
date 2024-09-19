from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model

@pipeline(enable_cache=False)
def train_pipeline(data_path:str):
    df=ingest_df(data_path)
    xtrain,xtest,ytrain,ytest=clean_df(df)
    model=train_model(xtrain,xtest,ytrain,ytest)

    mse,r2,rmse=evaluate_model(model,xtest,ytest)
    


