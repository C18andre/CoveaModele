################################################ main ################################################

# Imports externes
import pandas as pd
import warnings
warnings.simplefilter('ignore')

# Imports internes
from Projet.Config.Hyperparameters import getArgs
from Projet.Preprocessing.Conversion import configData
from Projet.Preprocessing.Standardisation import usefulCol,standardData
from Projet.Preprocessing.Batch import getRandomSample
from Projet.Modeles.Networks import LstmNet

# Import des HyperParamètres
args = getArgs()

# TODO : Suppression des ventes de moins de 10000
# TODO : Choix des données par ville, plus de random_batch
# TODO : Vérification de la taille du sample
# TODO : Enregistrement du sample sous le nom de la ville (upper())
# TODO : Enregistrer le modèle avec le nom de la ville (upper())
# TODO : Analyser l'importance de la normalisation
# TODO : Faire XGBoost et RandomForestRegressor

# Conversion et configuration de la donnée brute
def __config__() :  
    configData(args) # Cette fonction peut crasher facilement étant donné la quantité de donnée

# Standardisation de la donnnée
def __standard__() :
    usefulCol(args)    # Garder uniquement les colonnes que j'ai choisi
    standardData(args) # Remplacer les Na par des 0, 

# Obtention d'un échantillon aléatoire et conversion en float
def __sample__() :
    getRandomSample(args) # Problème avec les codes départements ()

# Select 

def __select__(ville) :
    # Check du type de donnée
    if type(ville) != str :
        raise ValueError("This is not a string")
    # Mis en majuscule si ce n'était pas le cas   
    ville = ville.upper()


# Train LSTM Net on the sample data
def __trainLSTM__() :
    data = pd.read_csv(args.path_sample_csv)
    LSTM = LstmNet(args,data)
    LSTM.train()
