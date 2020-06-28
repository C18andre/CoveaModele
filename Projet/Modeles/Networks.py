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

# Utilisation d'un LSTM pour prédire la valeur foncière d'une habitation
class LstmNet() :

    def __init__(self,args,data) :
        self.args = args
        self.data_train,self.data_validation,self.data_test = splitTVT(data,args)
        self.X_train,self.Y_train = splitXY(self.data_train)
        self.X_validation,self.Y_validation = splitXY(self.data_validation)
        self.X_test,self.Y_test = splitXY(self.data_test)
        self.optimizer = Adam(lr = self.args.learning_rate_lstm)
        self.trained = False
        
        # Données scalées
        if self.args.MinMax_scaler :
            self.X_scaler = MinMaxScaler()
            self.Y_scaler = MinMaxScaler()
        else :
            self.X_scaler = StandardScaler()
            self.Y_scaler = StandardScaler()
        
        self.X_scaled_train,self.Y_scaled_train = self.scaleXY(self.X_train,self.Y_train)
        self.X_scaled_validation,self.Y_scaled_validation = self.scaleXY(self.X_validation,self.Y_validation,initialize=False)
        self.X_scaled_test,self.Y_scaled_test = self.scaleXY(self.X_test,self.Y_test,initialize=False)

        # Load modele
        try :
            self.modele = load_model(self.args.save_lstm_path)
            print("Le modèle a bien été chargé")
            self.trained = True
        except :
            self.modele = self.createModele()

        # Log Paramètres
        self.logParams()
    
    # Création du modèle
    def createModele(self) :
        modele = Sequential()
        modele.add(LSTM(self.args.nb_1_layer, input_shape=(self.X_scaled_train.shape[1],self.X_scaled_train.shape[2]), return_sequences=False))
        modele.add(Dropout(self.args.dropout)) 
        modele.add(Dense(self.args.nb_2_layer, kernel_initializer='uniform', activation='elu'))
        modele.add(Dense(1, kernel_initializer='uniform', activation='elu'))
        modele.compile(loss='mse', optimizer= self.optimizer , metrics=['mae'])
        return modele

    # Training the model
    def train(self) :
        nb_epochs = self.args.nb_lstm_epochs
        batch_size = self.args.batch_lstm_size
        checkpoint = ModelCheckpoint(self.args.save_lstm_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        history = self.modele.fit(self.X_scaled_train,self.Y_scaled_train,epochs=nb_epochs, batch_size=batch_size, callbacks=callbacks_list,
                                validation_data=(self.X_scaled_validation,self.Y_scaled_validation),verbose = 1)
        self.trained = True
        self.saveResults(history)
        
    # Testing the model
    def test(self) :
        if not self.trained :
            print('The model is not trained')
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
        Y_scaled = Y_scaled.reshape(Y_scaled.shape[0],Y_scaled.shape[1])
        return X_scaled,Y_scaled
    
    def logParams(self) :
        self.modele.summary()
        print("")
        print(" Paramètres : ")
        print("")
        print("Shape train data = {}".format(self.X_scaled_train.shape))
        print("Shape validation data = {}".format(self.X_scaled_validation.shape))
        print("Shape test data = {}".format(self.X_scaled_test.shape))
        print("Learning rate = {}".format(self.args.learning_rate_lstm))
        print("")
    
    def saveResults(self,history) :
        df = pd.DataFrame()
        df['loss'] = history.history['loss']
        df['mae'] = history.history['mae']
        df['val_loss'] = history.history['val_loss']
        df['val_mae'] = history.history['val_mae']
        df.to_csv(self.args.path_results_lstm.format(self.args.nb_1_layer,self.args.nb_2_layer),index = False)
        