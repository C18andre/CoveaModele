# Imports externes
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Get a random sample from the dataframe
def getRandomSample(args) :
    data = pd.read_csv(args.path_data_csv)
    sample = data.sample(n = args.sample_size)
    sample.reset_index(inplace = True)
    del sample['index']
    sample_data = convertToFloat(sample)
    sample_data.to_csv(args.path_sample_csv,index = False)

# Convert to float
def convertToFloat(dataframe) :
    data = dataframe.copy()
    for column in data.columns :
        if data[column].dtypes == 'object' :
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
    dataframe.reset_index(inplace = True)
    del dataframe['index']
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



    
    

    