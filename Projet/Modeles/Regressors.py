# Régresseurs

# Imports externes
import numpy as np
import pandas as pd 
import sklearn
from sklearn.metrics import r2_score
import xgboost as xgb
from joblib import load,dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
# Imports internes
from Projet.Preprocessing.Batch import splitTVT,splitXY
from Projet.Preprocessing.Batch import getCitySample,normalize

#  Class XGB
class XgbReg() :
    def __init__(self,args,ville,code_departement) :
        
        # Get HP
        self.args = args
        self.trained = False
        # Loading data
        try :
            self.data = pd.read_csv(self.args.path_training_data_csv.format(ville.upper(),code_departement))
            print("")
            print("Data already exists and has sucessfully been loaded")
            print("")
        except :
            print("")
            print("Data has to be extract")
            print("")
            getCitySample(self.args,ville,code_departement)
            normalize(args,ville,code_departement)
            self.data = pd.read_csv(self.args.path_training_data_csv.format(ville.upper(),code_departement))
        
        # Path
        self.path = self.args.save_xgb_path.format(ville.upper(),code_departement)
        self.results_path = self.args.path_results_xgb.format(ville.upper(),code_departement)
        self.test_path = self.args.path_test_xgb.format(ville.upper(),code_departement)

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
            self.trained = True
        except :
            self.regressor = xgb.XGBRegressor(colsample_bytree=0.2, gamma=0.0, 
                             learning_rate=0.05, max_depth=6, 
                             min_child_weight=1.6, n_estimators=600,
                             reg_alpha=0.9, reg_lambda=0.6,
                             subsample=0.2, silent=1)
            print("")
            print("A new modele has been created")
            print("")
        
    # Entrainement du modele
    def train(self) :
        eval_result = {}
        callBacks = xgb.callback.record_evaluation(eval_result)
        self.regressor = self.regressor.fit(self.X_train,self.Y_train,eval_set = [(self.X_validation,self.Y_validation),(self.X_train,self.Y_train)],
                                            verbose=False,callbacks=[callBacks],eval_metric=['rmse','mae'])
        self.saveResults(eval_result)
        self.trained = True
        dump(self.regressor,self.path)

    # Test du modèle
    def test(self) :
        if not self.trained :
            print('The model is not trained')
        else :
            Y_pred = self.regressor.predict(self.X_test)
            Y_true = np.squeeze(self.Y_test)
            Y_pred = self.power(Y_pred)
            Y_true = self.power(Y_true)
            self.saveTest(Y_pred,Y_true)

    # Saving Results
    def saveResults(self,dictio) :
        data = pd.DataFrame()
        data['val_mse'] = dictio['validation_0']['rmse']
        data['mse'] = dictio['validation_1']['rmse']
        data['val_mae'] = dictio['validation_0']['mae']
        data['mae'] = dictio['validation_1']['mae']
        data.to_csv(self.results_path,index = False)

    # Save test results
    def saveTest(self,Y_pred,Y_true) :
        df = pd.DataFrame()
        df['Y_pred'] = Y_pred
        df['Y_true'] = Y_true
        df.to_csv(self.test_path,index = False)

    # Apply exp(x)-1 à un vecteur
    def power(self,Y) :
        Y_descale = []
        for i in range(len(Y)) :
            Y_descale.append(np.exp(Y[i])-1)
        return Y_descale






#  Class Random Forest Regressor

class RfrReg() :
    def __init__(self,args,ville,code_departement) :
        
        # Get HP
        self.args = args
        self.trained = False
        # Loading data
        try :
            self.data = pd.read_csv(self.args.path_training_data_csv.format(ville.upper(),code_departement))
            print("")
            print("Data already exists and has sucessfully been loaded")
            print("")
        except :
            print("")
            print("Data has to be extract")
            print("")
            getCitySample(self.args,ville,code_departement)
            normalize(args,ville,code_departement)
            self.data = pd.read_csv(self.args.path_training_data_csv.format(ville.upper(),code_departement))
        
        # Path
        self.path = self.args.save_rfr_path.format(ville.upper(),code_departement)
        self.results_path = self.args.path_results_rfr.format(ville.upper(),code_departement)
        self.test_path = self.args.path_test_rfr.format(ville.upper(),code_departement)

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
            self.trained = True

        except :
            self.regressor = RandomForestRegressor(n_estimators=150)
            print("")
            print("A new modele has been created")
            print("")

    # Entrainement du modele
    def train(self) :

        # Dictionaire d'hyperparamètres
        paramGridRFR = { "max_depth": [30,50], "min_samples_split": [5, 10, 20]}

        # Cherche la meilleur optimisation
        gridSearchRFR = GridSearchCV(self.regressor, paramGridRFR, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        gridSearchRFR.fit(self.X_train, self.Y_train)

        # On garde le meilleur
        self.regressor = gridSearchRFR.best_estimator_
        self.trained = True
        # Mae sur validation
        Y_pred = self.regressor.predict(self.X_validation)
        mae = mean_absolute_error(self.Y_validation,Y_pred)
        self.saveResults([mae])
        
        # Sauvegarde
        dump(self.regressor,self.path)


    # Test du modèle
    def test(self) :
        if not self.trained :
            print('The model is not trained')
        else :
            Y_pred = self.regressor.predict(self.X_test)
            Y_true = np.squeeze(self.Y_test)
            Y_pred = self.power(Y_pred)
            Y_true = self.power(Y_true)
            self.saveTest(Y_pred,Y_true)

    # Saving Results
    def saveResults(self,val) :
        data = pd.DataFrame()
        data['val_mae'] = val
        data.to_csv(self.results_path,index = False)

    # Save test results
    def saveTest(self,Y_pred,Y_true) :
        df = pd.DataFrame()
        df['Y_pred'] = Y_pred
        df['Y_true'] = Y_true
        df.to_csv(self.test_path,index = False)

    # Apply exp(x)-1 à un vecteur
    def power(self,Y) :
        Y_descale = []
        for i in range(len(Y)) :
            Y_descale.append(np.exp(Y[i])-1)
        return Y_descale