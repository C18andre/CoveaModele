################################################ main ################################################

# Imports externes
import pandas as pd

# Imports internes
from Projet.Config.Hyperparameters import getArgs
from Projet.Preprocessing.Conversion import configData
from Projet.Preprocessing.Standardisation import usefulCol,standardData

# Import des HyperParamètres
args = getArgs()

# Conversion et configuration de la donnée brute
def __config__() :  
    configData(args) # Cette fonction peut crasher facilement étant donné la quantité de donnée

# Standardisation de la donnnée
def __standard__() :
    usefulCol(args)
    standardData(args)

__standard__()