# Import externe
import argparse

# Hyperparamètres du projet

def getArgs() :
    parser = argparse.ArgumentParser(description='HyperParameters')
    parser.add_argument('--path_txt',type = str,default = 'Projet/Data/valeursfoncieres-2019.txt')
    parser.add_argument('--path_raw_csv',type = str,default = 'Projet/Data/raw_data.csv')
    parser.add_argument('--path_clean_csv',type = str,default='Projet/Data/clean_data.csv')
    parser.add_argument('--path_useful_data',type = str, default ='Projet/Data/useful_data.csv')
    parser.add_argument('--colonnes_utiles',type = list,default = ['Valeur fonciere','Type de voie','Code departement','No Volume','Nombre de lots','Code type local',
                        'Surface reelle bati','Nombre pieces principales','Code postal','Surface terrain'])                     
    parser.add_argument('--f')
    args = parser.parse_args()
    return args
