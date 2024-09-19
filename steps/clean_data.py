import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning,DataDivideStrategy,DataPreprocessingStrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_df(df:pd.DataFrame)->Tuple[Annotated[pd.DataFrame,"xtrain"],
                                     Annotated[pd.DataFrame,"xtest"],
                                     Annotated[pd.Series,"ytrain"],
                                     Annotated[pd.Series,"ytest"]]:
    try:
        process_strategy=DataPreprocessingStrategy()
        data_cleaning=DataCleaning(df,process_strategy)
        processed_data=data_cleaning.handle_data()

        divide_strategy=DataDivideStrategy()
        data_cleaning=DataCleaning(processed_data,divide_strategy)
        xtrain,xtest,ytrain,ytest=data_cleaning.handle_data()
        logging.info("data cleaning comleted")
        return xtrain,xtest,ytrain,ytest
    except Exception as e:
        logging.error("Error in cleaning data:{}".format(e))
        raise e
