# Imports externes
import tensorflow as tf 
import pandas as pd 
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation, Input, LSTM, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

# Imports internes
from Projet.Preprocessing.Batch import splitTVT,splitXY
from Projet.Preprocessing.Batch import getCitySample,normalize

# Utilisation d'un LSTM pour prédire la valeur foncière d'une habitation
class NeuralNet() :

    def __init__(self,args,ville,code_departement) :

        # Get HP
        self.args = args

        # Loading data
        try :
            self.data = pd.read_csv(self.args.path_training_data_csv.format(ville.upper(),code_departement))
            print("")
            print("Data already exists and has sucessfully been loaded")
            print("")
        except :
            print("")
            print("Data has to be extract")
            print("")
            getCitySample(self.args,ville,code_departement)
            normalize(args,ville,code_departement)
            self.data = pd.read_csv(self.args.path_training_data_csv.format(ville.upper(),code_departement))
        
        # Paths
        self.path = self.args.save_nn_path.format(ville.upper(),code_departement)
        self.results_path = self.args.path_results_nn.format(ville.upper(),code_departement)
        self.test_path = self.args.path_test_nn.format(ville.upper(),code_departement)

        # Data
        self.data_train,self.data_validation,self.data_test = splitTVT(self.data,self.args)
        self.X_train,self.Y_train = splitXY(self.data_train)
        self.X_validation,self.Y_validation = splitXY(self.data_validation)
        self.X_test,self.Y_test = splitXY(self.data_test)
        self.optimizer = Adam(lr = self.args.learning_rate_nn)
        self.trained = False
        
        self.X_train,self.Y_train = self.reshape(self.X_train,self.Y_train)
        self.X_validation,self.Y_validation = self.reshape(self.X_validation,self.Y_validation)
        self.X_test,self.Y_test = self.reshape(self.X_test,self.Y_test)

        # Load modele
        try :
            self.modele = load_model(self.path)
            print("")
            print("Le modèle pour la ville de {} a bien été chargé ".format(ville))
            print("")
            self.trained = True
        except :
            print("")
            print("A new modele has been created")
            print("")
            self.modele = self.createModele()

        # Log Paramètres
        self.logParams()
    
    # Création du modèle
    def createModele(self) :
        modele = Sequential()
        modele.add(Dense(self.args.nb_1_layer, input_shape=(self.X_train.shape[1],self.X_train.shape[2]), 
                        kernel_initializer='uniform', activation='elu'))
        modele.add(Dense(self.args.nb_2_layer, kernel_initializer='uniform', activation='elu'))
        modele.add(Dense(1, kernel_initializer='uniform', activation='elu'))
        modele.compile(loss='mse', optimizer= self.optimizer , metrics=['mae'])
        return modele

    # Training the model
    def train(self) :
        nb_epochs = self.args.nb_nn_epochs
        batch_size = self.args.batch_nn_size
        checkpoint = ModelCheckpoint(self.path, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        history = self.modele.fit(self.X_train,self.Y_train,epochs=nb_epochs, batch_size=batch_size, callbacks=callbacks_list,
                            validation_data=(self.X_validation,self.Y_validation),verbose = 1)
        
        self.trained = True
        self.saveResults(history)
        
    # Testing the model
    def test(self) :
        if not self.trained :
            print('The model is not trained')
        else :
            Y_pred = self.modele.predict(self.X_test)
            Y_pred = tf.squeeze(Y_pred).numpy()
            Y_pred = self.power(Y_pred)
            Y_test = np.squeeze(self.Y_test)
            Y_true = self.power(Y_test)
            self.saveTest(Y_pred,Y_true)


    # Affiche les paramètres
    def logParams(self) :
        self.modele.summary()
        print("")
        print(" Paramètres : ")
        print("")
        print("Shape train data = {}".format(self.X_train.shape))
        print("Shape validation data = {}".format(self.X_validation.shape))
        print("Shape test data = {}".format(self.X_test.shape))
        print("Learning rate = {}".format(self.args.learning_rate_nn))
        print("")
    
    # Sauvegarde des résultats
    def saveResults(self,history) :
        df = pd.DataFrame()
        df['loss'] = history.history['loss']
        df['mae'] = history.history['mae']
        df['val_loss'] = history.history['val_loss']
        df['val_mae'] = history.history['val_mae']
        df.to_csv(self.results_path,index = False)
    
    # Save test results
    def saveTest(self,Y_pred,Y_true) :
        df = pd.DataFrame()
        df['Y_pred'] = Y_pred
        df['Y_true'] = Y_true
        df.to_csv(self.test_path,index = False)

    # Reshape pour le modèle
    def reshape(self,X,Y) :
        X = X.reshape(X.shape[0],1,X.shape[1])
        Y = Y.reshape(Y.shape[0],1,Y.shape[1])
        return X,Y
    
    # Apply exp(x)-1 à un vecteur
    def power(self,Y) :
        Y_descale = []
        for i in range(len(Y)) :
            Y_descale.append(np.exp(Y[i])-1)
        return Y_descale