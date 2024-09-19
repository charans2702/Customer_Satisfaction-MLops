import logging
from abc import ABC,abstractmethod
from sklearn.linear_model import LinearRegression
class Model(ABC):

    @abstractmethod
    def train(self,xtrain,ytrain):
        pass

class LinearRegressionModel(Model):

    def train(self,xtrain,ytrain):

        try:
            reg=LinearRegression()
            reg.fit(xtrain,ytrain)
            logging.info("model training comleted")
            return reg
        except Exception as e:
            logging.error("error in training model:{}".format(e))
            raise e