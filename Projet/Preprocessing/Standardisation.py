# Imports externes
import pandas as pd


# Créer un csv avec les colonnes utiles pour la prédiction
def usefulCol(args) :
    data = pd.read_csv(args.path_clean_csv)
    columns = args.colonnes_utiles
    new_data = data[columns]
    new_data.to_csv(args.path_useful_csv)


# Standardisation

def standardData(args) :
    