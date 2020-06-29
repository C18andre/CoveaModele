################################################ main ################################################

# Imports externes
import pandas as pd
import warnings
warnings.simplefilter('ignore')

# Imports internes
from Projet.Config.Hyperparameters import getArgs
from Projet.Preprocessing.Conversion import configData
from Projet.Preprocessing.Batch import getCitySample,normalize
from Projet.Modeles.Networks import NeuralNet
from Projet.Modeles.Regressors import XgbReg

# Import des HyperParamètres
args = getArgs()

# TODO : Faire XGBoost et RandomForestRegressor

# Configuration de la donnée brute || __config__()
def __config__() :  
    configData(args) # Cette fonction peut crasher facilement étant donné la quantité de donnée

# Obtention des données (Utile pour l'analyse des données)|| Exemple : __select__('Marseille',13)
def __select__(ville,code_departement) :
    getCitySample(args,ville,code_departement)
    normalize(args,ville,code_departement)

# Entrainement du réseau de neurones pour une ville en particulier|| Exemple : __trainNN__('Marseille',13)
def __trainNN__(ville,code_departement) :
    NN = NeuralNet(args,ville,code_departement)
    NN.train()

# Entrainement du XGB pour une ville en particulier || Exemple : __trainXGB__('Marseille',13)
def __trainXGB__(ville,code_departement) :
    XGB = XgbReg(args,ville,code_departement)
    XGB.train()

__trainXGB__('Lyon',69)