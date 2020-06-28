################################################ main ################################################

# Imports externes
import pandas as pd
import warnings
warnings.simplefilter('ignore')

# Imports internes
from Projet.Config.Hyperparameters import getArgs
from Projet.Preprocessing.Conversion import configData
from Projet.Preprocessing.Batch import getCitySample
from Projet.Modeles.Networks import LstmNet

# Import des HyperParamètres
args = getArgs()

# TODO : Enregistrer le modèle avec le nom de la ville (upper())
# TODO : Analyser l'importance de la normalisation
# TODO : Faire XGBoost et RandomForestRegressor

# Configuration de la donnée brute
def __config__() :  
    configData(args) # Cette fonction peut crasher facilement étant donné la quantité de donnée

# Selectionne les données en fonction d'une ville
# Réalise également le tri (Pas de transactions à moins de 10000, uniquement les colonnes utiles) et Conversion en float 
def __select__(ville,code_departement) :
    getCitySample(args,ville,code_departement)

# Train LSTM Net on the sample data
def __trainLSTM__() :
    data = pd.read_csv(args.path_sample_csv)
    LSTM = LstmNet(args,data)
    LSTM.train()

__select__('Toulouse',31)
