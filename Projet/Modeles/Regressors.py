# Régresseurs

# Imports externes
import numpy as np
import pandas as pd 
import sklearn
from sklearn.metrics import r2_score
import xgboost as xgb
from joblib import load,dump

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

        # Data
        self.data_train,self.data_validation,self.data_test = splitTVT(self.data,self.args)
        self.X_train,self.Y_train = splitXY(self.data_train)
        self.X_validation,self.Y_validation = splitXY(self.data_validation)
        self.X_test,self.Y_test = splitXY(self.data_test)

        # Load modele
        try :
            self.regressor = load(self.path)
            print("")
            print("Modele has sucessfully been loaded")
            print("")
        except :
            self.regressor = xgb.XGBRegressor(gamma=0.0,n_estimators=150,base_score=0.7,colsample_bytree=1,learning_rate=0.05)
            print("")
            print("A new modele has been created")
            print("")
        
    # Entrainement du modele
    def train(self) :
        eval_result = {}
        callBacks = xgb.callback.record_evaluation(eval_result)
        self.regressor = self.regressor.fit(self.X_train,self.Y_train,eval_set = [(self.X_validation,self.Y_validation)],verbose=True,callbacks=[callBacks])
        print(eval_result)

    
        





