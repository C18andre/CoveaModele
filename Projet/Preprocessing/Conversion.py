# Imports externes
import pandas as pd

# Convertir en fichier csv
def __convert__(load_path,save_path) :
    # Lecture
    try : 
        data = pd.read_csv(load_path,delimiter='|')
    except :
        print(" The path is not correct or the file does not exist")
        print(load_path)
    data.to_csv(save_path,index = False)

# Supprimer les colonnes vides
def __deleteCol__(load_path,save_path) :
    # Lecture
    try :
        data = pd.read_csv(load_path)
    except :
        print(' The path is not correct or the file does not exist')
        print(load_path)

    data.dropna(axis = 1,how = 'all',inplace = True)
    data.to_csv(save_path,index = False)

# Filtrer pour garder les ventes et les maisons/appartements uniquements
def __filter__(load_path,save_path) :
    # Lecture
    try :
        data = pd.read_csv(load_path)
    except :
        print(" The path is not correct or the file does not exist")
        print(load_path)
    
    # Maison et appartement
    data_ = data['Code type local'] <= 2.0 
    new_data = data.where(data_)

    # Vente uniquement
    data_vente = new_data['Nature mutation'] == 'Vente'
    clean_data = new_data.where(data_vente)

    # Supression des lignes vides
    clean_data.dropna(axis = 0,how = 'all',inplace = True)
    clean_data.to_csv(save_path,index = False)

# Configuration of Data
def configData(args) :
    __convert__(args.path_txt,args.path_raw_csv)
    __deleteCol__(args.path_raw_csv,args.path_raw_csv)
    __filter__(args.path_raw_csv,args.path_clean_csv)
