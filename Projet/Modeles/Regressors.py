# Régresseurs

# Imports externes
import numpy as np
import pandas as pd 
import sklearn
from sklearn.metrics import r2_score
import xgboost as xgb
from joblib import load,dump
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Imports internes
from Projet.Preprocessing.Batch import splitTVT,splitXY
from Projet.Preprocessing.Batch import getCitySample

#  Class XGB
class XgbReg() :
    def __init__(self,args,ville,code_departement) :
        
        # Get HP
        self.args = args

        # Loading data
        try :
            self.data = pd.read_csv(self.args.path_data_csv.format(ville.upper(),code_departement))
            print("")
            print("Data already exists and has sucessfully been loaded")
            print("")
        except :
            print("")
            print("Data has to be extract")
            print("")
            getCitySample(self.args,ville,code_departement)
            self.data = pd.read_csv(self.args.path_data_csv.format(ville.upper(),code_departement))
        
        # Path
        self.path = self.args.save_xgb_path.format(ville.upper(),code_departement)
        self.results_path = self.args.path_results_xgb.format(ville.upper(),code_departement)

        # Scaler
        if self.args.MinMax_scaler :
            self.X_scaler = MinMaxScaler()
            self.Y_scaler = MinMaxScaler()
        else :
            self.X_scaler = StandardScaler()
            self.Y_scaler = StandardScaler()

        # Data
        self.data_train,self.data_validation,self.data_test = splitTVT(self.data,self.args)
        self.X_train,self.Y_train = splitXY(self.data_train)
        self.X_validation,self.Y_validation = splitXY(self.data_validation)
        self.X_test,self.Y_test = splitXY(self.data_test)        
        
        # Scale data
        self.X_scaled_train,self.Y_scaled_train = self.scaleXY(self.X_train,self.Y_train)
        self.X_scaled_validation,self.Y_scaled_validation = self.scaleXY(self.X_validation,self.Y_validation,initialize=False)
        self.X_scaled_test,self.Y_scaled_test = self.scaleXY(self.X_test,self.Y_test,initialize=False)

        # Load modele
        try :
            self.regressor = load(self.path)
            print("")
            print("Modele has sucessfully been loaded")
            print("")
        except :
            self.regressor = xgb.XGBRegressor(colsample_bytree=0.2, gamma=0.0, 
                             learning_rate=0.05, max_depth=6, 
                             min_child_weight=1.5, n_estimators=100,
                             reg_alpha=0.9, reg_lambda=0.6,
                             subsample=0.2, silent=1)
            print("")
            print("A new modele has been created")
            print("")
        
    # Entrainement du modele
    def train(self) :
        eval_result = {}
        callBacks = xgb.callback.record_evaluation(eval_result)
        self.regressor = self.regressor.fit(self.X_scaled_train,self.Y_scaled_train,eval_set = [(self.X_scaled_validation,self.Y_scaled_validation)],verbose=True,callbacks=[callBacks])
        self.saveResults(eval_result)
        dump(self.regressor,self.path)

    # Scale the data
    def scaleXY(self,X,Y,initialize = True) :
        # Fittage du scaler
        if initialize :
            self.X_scaler = self.X_scaler.fit(X)
            self.Y_scaler = self.Y_scaler.fit(Y)
        # Transformation
        X_scaled = self.X_scaler.transform(X)
        Y_scaled = self.Y_scaler.transform(Y)

        return X_scaled,Y_scaled

    # Saving Results
    def saveResults(self,dictio) :
        data = pd.DataFrame()
        data['validation'] = dictio['validation_0']['rmse']
        data.to_csv(self.results_path)




