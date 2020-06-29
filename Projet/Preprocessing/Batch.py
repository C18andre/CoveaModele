# Imports externes
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Get the data for a particular city
def getCitySample(args,ville,code_departement) :
    # Check du type de donnée
    if type(ville) != str :
        raise ValueError("This is not a string")
    if type(code_departement) == str :
        code_departement = int(code_departement)
    
    # Enregistrement des données
    try : 
        city_data = pd.read_csv(args.path_data_csv.format(ville,code_departement))
        print("Data already exist")
        
    except :
        # Mis en majuscule si ce n'était pas le cas   
        ville = ville.upper()

        # Get Clean Data
        clean_data = pd.read_csv(args.path_clean_csv)

        # Match city + code departement
        data_ = clean_data['Commune'].str.match(ville)
        city_data = clean_data.where(data_)
        city_data.dropna(inplace = True,how = 'all')
        data_ = city_data['Code departement'] == code_departement
        city_data = city_data.where(data_)
        city_data.dropna(inplace = True,how = 'all')

        # Choix des colonnes utiles et fillna(0)
        columns = args.colonnes_utiles
        city_data = city_data[columns]
        city_data.fillna(0,inplace = True)
        city_data.reset_index(inplace = True,drop = True)

        # Convertir les colonnes Type de voie et Code commmune
        le_cc = LabelEncoder()
        city_data['Code commune'].fillna(0,inplace = True)
        le_cc.fit(city_data['Code commune'].to_list())
        city_data['Code commune'] = le_cc.transform(city_data['Code commune'].to_list())

        le_tv = LabelEncoder()
        city_data['Type de voie'].fillna(0,inplace = True)
        le_tv.fit(city_data['Type de voie'].to_list())
        city_data['Type de voie'] = le_tv.transform(city_data['Type de voie'].to_list())      
        
        # Convertir tout en float
        city_data = convertToFloat(city_data)
    
    # Supprimer les lignes avec des transactions de moins de 10000
    data_ = city_data['Valeur fonciere'] >= args.minimum_transaction
    clean_city_data = city_data.where(data_)
    clean_city_data.dropna(inplace = True,how = 'all')
    
    # Supprimer les transactions de plus de 1 500 000€
    data_ = clean_city_data['Valeur fonciere'] <= args.maximum_transaction
    clean_city_data = clean_city_data.where(data_)
    clean_city_data.dropna(inplace = True,how = 'all')

    # Suppression des doublons
    clean_city_data.drop_duplicates(inplace = True)

    # Check the len of the data
    if len(clean_city_data) < args.minimum_size :
        print("Not enough data")
        print("Len = {}".format(len(clean_city_data)))

    # Sauvergarde des données
    clean_city_data.to_csv(args.path_data_csv.format(ville,code_departement),index = False)


# Convert to float
def convertToFloat(dataframe) :
    data = dataframe.copy()
    for column in data.columns :
        if data[column].dtypes == 'object':
            for i in range(len(data)) :
                if type(data[column][i]) == str :
                        try :
                            data[column][i] = float(data[column][i].replace(',','.'))
                        except :
                            data[column][i] = 0.0
    return data

# Split Train-Validation-Test
def splitTVT(data,args) :
    split_train = int(len(data)*args.taille_train)
    split_validation = int(len(data)*args.taille_validation)
    train = data[: split_train]
    validation = data[split_train : split_train + split_validation ]
    test = data[split_train + split_validation : :]
    return train,validation,test

# Split X/Y
def splitXY(data) :
    dataframe = data.copy()
    # Supression de l'index
    dataframe.reset_index(inplace = True,drop = True)
    # Colonne target
    data_target = dataframe['Valeur fonciere']
    del dataframe['Valeur fonciere']
    # Création des vecteurs X et Y
    dataX, dataY = [], []
    for i in range(len(dataframe)):
        a = dataframe.iloc[i]
        dataX.append(a)
        dataY.append([data_target[i]])
    return np.array(dataX), np.array(dataY)



    
    

    