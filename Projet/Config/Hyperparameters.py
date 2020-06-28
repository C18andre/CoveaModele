# Import externe
import argparse

# Hyperparamètres du projet

def getArgs() :

    # Configuration des données
    parser = argparse.ArgumentParser(description='HyperParameters')
    parser.add_argument('--path_txt',type = str,default = 'Projet/Data/valeursfoncieres-2019.txt')
    parser.add_argument('--path_raw_csv',type = str,default = 'Projet/Data/raw_data.csv')
    parser.add_argument('--path_clean_csv',type = str,default='Projet/Data/clean_data.csv')
    parser.add_argument('--path_useful_csv',type = str, default ='Projet/Data/useful_data.csv')
    parser.add_argument('--path_data_csv',type = str, default ='Projet/Data/data.csv')
    parser.add_argument('--colonnes_utiles',type = list,default = ['Valeur fonciere','Code departement','Nombre de lots','Code type local',
                        'Surface reelle bati','Nombre pieces principales','Code commune','Surface terrain','1er lot','Surface Carrez du 1er lot',
                        '2eme lot','Surface Carrez du 2eme lot','3eme lot','Surface Carrez du 3eme lot','4eme lot','Surface Carrez du 4eme lot',
                        '5eme lot','Surface Carrez du 5eme lot'])
    
    # HP pour la taille de l'échantillon
    parser.add_argument('--sample_size',type = int,default = 10000)  
    parser.add_argument('--path_sample_csv',type = str, default ='Projet/TrainingData/sample.csv')          
    parser.add_argument('--taille_train',type = float,default = 0.8)
    parser.add_argument('--taille_validation',type = float,default = 0.1)

    # Paramètres pour le LSTM
    parser.add_argument('--nb_1_layer',type = int,default = 60)  
    parser.add_argument('--nb_2_layer',type = int,default = 30)
    parser.add_argument('--learning_rate_lstm',type = float,default = 0.001)
    parser.add_argument('--dropout',type = float, default = 0.05)
    parser.add_argument('--nb_lstm_epochs',type = int,default = 100)
    parser.add_argument('--batch_lstm_size',type = int,default = 50)
    parser.add_argument('--save_lstm_path',type = str,default = 'Projet/Modeles/LSTM_modele.h5')
    parser.add_argument('--MinMax_scaler',type = bool,default = True)
    parser.add_argument('--f')
    args = parser.parse_args()
    return args
