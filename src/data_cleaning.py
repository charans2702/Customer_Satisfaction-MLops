import logging
import pandas as pd
from abc import ABC,abstractmethod

from typing import Union
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class DataStrategy(ABC):

    @abstractmethod
    def handle_data(self,data:pd.DataFrame)->Union[pd.DataFrame,pd.Series]:
        pass

class DataPreprocessingStrategy(DataStrategy):

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        
        try:
            data=data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                    'order_id',
                    'customer_id',
                    'customer_unique_id',
                    'customer_zip_code_prefix',
                    'order_item_id',
                    'product_id',
                    'seller_id',
                    'review_comment_message',
                    'product_category_name',
                    'product_category_name_english',
                    'shipping_limit_date',
                    'customer_city',
                    'customer_state'
                ],
                axis=1
            )
            data['product_weight_g'].fillna(data['product_weight_g'].median(),inplace=True)
            data['product_length_cm'].fillna(data['product_length_cm'].median(),inplace=True)
            data['product_height_cm'].fillna(data['product_height_cm'].median(),inplace=True)
            data['product_width_cm'].fillna(data['product_width_cm'].median(),inplace=True)
            onehot_categories=['order_status','payment_type']
            onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            onehot_encoded = onehot_encoder.fit_transform(data[onehot_categories])
            onehot_columns = onehot_encoder.get_feature_names_out(onehot_categories)
            df_onehot = pd.DataFrame(onehot_encoded, columns=onehot_columns, index=data.index)

            data= pd.concat([data.drop(columns=onehot_categories), df_onehot], axis=1)
            return data
        except Exception as e:
            logging.error("Error in preprocessing data:{}".format(e))
            raise e
class DataDivideStrategy(DataStrategy):

    def handle_data(self,data:pd.DataFrame):

        try:
            x=data.drop(["review_score"],axis=1)
            y=data["review_score"]
            xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
            return xtrain,xtest,ytrain,ytest
        except Exception as e:
            logging.error("Error in dividing data:{}".format(e))
            raise e
class DataCleaning:

    def __init__(self,data:pd.DataFrame,strategy:DataStrategy):
        self.data=data
        self.strategy=strategy

    def handle_data(self)->Union[pd.DataFrame,pd.Series]:

        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data:{}".format(e))
            raise e

           


            

