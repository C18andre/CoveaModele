################################################ main ################################################

# Imports externes
import pandas as pd

# Imports internes
from Projet.Config.Hyperparameters import getArgs
from Projet.Preprocessing.Conversion import configData


# Import des HyperParamètres
args = getArgs()

# Conversion et configuration de la donnée brute
def __config__() :  
    configData(args) # Cette fonction peut crasher facilement étant donné la quantité de donnée

