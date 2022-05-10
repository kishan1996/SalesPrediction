import EDA.preprocessing as p
from sklearn.ensemble import RandomForestRegressor
import pickle
from app_logging.logging import CustomLogger
from exception.exception import CustomException
logger = CustomLogger("logs")
from utils.utils import read_config
import sys
from sklearn.metrics import mean_absolute_error



class train:
    
    def __init__(self):
       self.config = read_config()
       print(self.config)
       self.n_estimators = self.config["model_parameters"]["n_estimators"]
       self.max_depth = self.config["model_parameters"]["max_depth"]
       self.max_features = self.config["model_parameters"]["max_features"]
       self.min_samples_split = self.config["model_parameters"]["min_samples_split"]
       self.min_samples_leaf = self.config["model_parameters"]["min_samples_leaf"]
       
   
    def rf(self,X_train,Y_train):
        try:
            self.X_train = X_train
            self.Y_train = Y_train
            print(self.n_estimators,  self.max_depth,  self.max_features,  self.min_samples_split,  self.min_samples_leaf)
            RF = RandomForestRegressor(n_estimators=self.n_estimators,  max_depth=self.max_depth,  max_features=self.max_features,  min_samples_split=self.min_samples_split,  min_samples_leaf=self.min_samples_leaf)
            RF.fit(self.X_train,  self.Y_train)
            model_save_file = open("SavedModel.sav", 'wb')
            pickle.dump(RF, model_save_file)
            model_save_file.close()
          


            print("Training Complete :)")
            


        except Exception as e:
            message = CustomException(e, sys)
            logger.error(message.error_message)
        




if __name__ == "__main__":

        dataloading = p.Preprocessing()
        train_detail,test_detail = dataloading.dataloading()
        X_train,Y_train = dataloading.drop1(train_detail,test_detail)

        train1=train()
        train1.rf(X_train,Y_train)
  
   
        