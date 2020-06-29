# Imports externes
import tensorflow as tf 
import pandas as pd 
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation, Input, LSTM, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Imports internes
from Projet.Preprocessing.Batch import splitTVT,splitXY
from Projet.Preprocessing.Batch import getCitySample

# Utilisation d'un LSTM pour prédire la valeur foncière d'une habitation
class NeuralNet() :

    def __init__(self,args,ville,code_departement) :

        # Get HP
        self.args = args

        # Loading data
        try :
            self.data = pd.read_csv(self.args.path_data_csv.format(ville.upper(),code_departement))
            print("")
            print("Data already exists and has sucessfully been loaded")
            print("")
        except :
            print("")
            print("Data has to be extract")
            print("")
            getCitySample(self.args,ville,code_departement)
            self.data = pd.read_csv(self.args.path_data_csv.format(ville.upper(),code_departement))
        
        # Paths
        self.path = self.args.save_nn_path.format(ville.upper(),code_departement)
        self.results_path = self.args.path_results_nn.format(ville.upper(),code_departement)

        # Data
        self.data_train,self.data_validation,self.data_test = splitTVT(self.data,self.args)
        self.X_train,self.Y_train = splitXY(self.data_train)
        self.X_validation,self.Y_validation = splitXY(self.data_validation)
        self.X_test,self.Y_test = splitXY(self.data_test)
        self.optimizer = Adam(lr = self.args.learning_rate_nn)
        self.trained = False
        
        # Données scalées
        if self.args.MinMax_scaler :
            self.X_scaler = MinMaxScaler()
            self.Y_scaler = MinMaxScaler()
        else :
            self.X_scaler = StandardScaler()
            self.Y_scaler = StandardScaler()
        
        # Scale data
        self.X_scaled_train,self.Y_scaled_train = self.scaleXY(self.X_train,self.Y_train)
        self.X_scaled_validation,self.Y_scaled_validation = self.scaleXY(self.X_validation,self.Y_validation,initialize=False)
        self.X_scaled_test,self.Y_scaled_test = self.scaleXY(self.X_test,self.Y_test,initialize=False)

        # Reshape 
        if not self.args.normalize :
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
        modele.add(Dense(self.args.nb_1_layer, input_shape=(self.X_scaled_train.shape[1],self.X_scaled_train.shape[2]), 
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
        if self.args.normalize :
            history = self.modele.fit(self.X_scaled_train,self.Y_scaled_train,epochs=nb_epochs, batch_size=batch_size, callbacks=callbacks_list,
                                validation_data=(self.X_scaled_validation,self.Y_scaled_validation),verbose = 1)
        else :
            history = self.modele.fit(self.X_train,self.Y_train,epochs=nb_epochs, batch_size=batch_size, callbacks=callbacks_list,
                                validation_data=(self.X_validation,self.Y_validation),verbose = 1)
        
        self.trained = True
        self.saveResults(history)
        
    # Testing the model
    def test(self) :
        if not self.trained :
            print('The model is not trained')
        else :
            if self.args.normalize :
                metrics = self.modele.evaluate(self.X_scaled_test,self.Y_scaled_test,verbose = 1)
            else :
                metrics = self.modele.evaluate(self.X_test,self.Y_test,verbose = 1)
    
    # Scale the data
    def scaleXY(self,X,Y,initialize = True) :
        # Fittage du scaler
        if initialize :
            self.X_scaler = self.X_scaler.fit(X)
            self.Y_scaler = self.Y_scaler.fit(Y)
        # Transformation
        X_scaled = self.X_scaler.transform(X)
        Y_scaled = self.Y_scaler.transform(Y)

        # Reshape 
        X_scaled = X_scaled.reshape(X_scaled.shape[0],1,X_scaled.shape[1])
        Y_scaled = Y_scaled.reshape(Y_scaled.shape[0],1,Y_scaled.shape[1])
        return X_scaled,Y_scaled
    
    # Affiche les paramètres
    def logParams(self) :
        self.modele.summary()
        print("")
        print(" Paramètres : ")
        print("")
        print("Shape train data = {}".format(self.X_scaled_train.shape))
        print("Shape validation data = {}".format(self.X_scaled_validation.shape))
        print("Shape test data = {}".format(self.X_scaled_test.shape))
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
    
    # Reshape pour le modèle
    def reshape(self,X,Y) :
        X = X.reshape(X.shape[0],1,X.shape[1])
        Y = Y.reshape(Y.shape[0],1,Y.shape[1])
        return X,Y

        