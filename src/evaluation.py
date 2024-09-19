import logging 
from abc import ABC,abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score

class Evaluation(ABC):

    @abstractmethod
    def calculate_scores(self,ytrue:np.ndarray,ypred:np.ndarray):
        pass
class MSE(Evaluation):
    def calculate_scores(self,ytrue:np.ndarray,ypred:np.ndarray):

        try:
            logging.info("Calculation MSE")
            mse=mean_squared_error(ytrue,ypred)
            logging.info("MSE:{}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculatnf MSE:{}".format(e))
            raise e

class R2(Evaluation):

    def calculate_scores(self,ytrue:np.ndarray,ypred:np.ndarray):
        try:
            logging.info("Calculating R2 score")
            r2=r2_score(ytrue,ypred)
            logging.info("R2 Score:{}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating r2 score:{}".format(e))
            raise e

class RMSE(Evaluation):

    def calculate_scores(self,ytrue:np.ndarray,ypred:np.ndarray):

        try:
            logging.info("Calculation RMSE")
            rmse=mean_squared_error(ytrue,ypred,squared=False)
            logging.info("RMSE:{}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in calculatnf RMSE:{}".format(e))
            raise e

