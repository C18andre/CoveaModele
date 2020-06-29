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
    parser.add_argument('--path_data_csv',type = str, default ='Projet/Data/{}_{}_data.csv')
    parser.add_argument('--path_training_data_csv',type = str,default = 'Projet/TrainingData/{}_{}_training_data.csv')
    parser.add_argument('--colonnes_utiles',type = list,default = ['Valeur fonciere','Nombre de lots','Type de voie','Code type local',
                        'Surface reelle bati','Nombre pieces principales','Surface terrain','Code commune','Surface Carrez du 1er lot',
                        'Surface Carrez du 2eme lot','Surface Carrez du 3eme lot','Surface Carrez du 4eme lot',
                        'Surface Carrez du 5eme lot'])
                        
    # HP pour la taille de l'échantillon
    parser.add_argument('--minimum_size',type = int,default = 1000)  
    parser.add_argument('--minimum_transaction',type = float,default=10000.0)  
    parser.add_argument('--maximum_transaction',type = float,default=1500000.0)  
    parser.add_argument('--taille_train',type = float,default = 0.8)
    parser.add_argument('--taille_validation',type = float,default = 0.1)

    # Paramètres pour le NeuralNet
    parser.add_argument('--nb_1_layer',type = int,default = 100)  
    parser.add_argument('--nb_2_layer',type = int,default = 40)
    parser.add_argument('--learning_rate_nn',type = float,default = 0.001)
    parser.add_argument('--nb_nn_epochs',type = int,default = 100)
    parser.add_argument('--batch_nn_size',type = int,default = 50)
    parser.add_argument('--save_nn_path',type = str,default = 'Projet/Modeles/PreTrained/NN_{}_{}.h5')
    parser.add_argument('--path_results_nn',type = str,default = 'Projet/Results/NN_results_{}_{}.csv')

    # Paramètres pour le XGB
    parser.add_argument('--save_xgb_path',type = str,default = 'Projet/Modeles/PreTrained/XGB_{}_{}.joblib')
    parser.add_argument('--path_results_xgb',type = str,default = 'Projet/Results/XGB_results_{}_{}.csv')

    args = parser.parse_args()
    return args
