from pyclbr import Class
import numpy as np #For algebrical operations.
import pandas as pd # For EDA purposes.
import seaborn as sns # For Datavisuals
import matplotlib.pyplot as plt #For Data visuals
import tensorflow as tf

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout

class RNNRegressor:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.build_model()
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def build_model(self):
        model = Sequential()

        # Add the first SimpleRNN layer and add dropout regularization
        model.add(SimpleRNN(units=50, activation="tanh", return_sequences=True, input_shape=self.input_shape))
        model.add(Dropout(0.2))

        # Add the second SimpleRNN layer and add dropout regularization
        model.add(SimpleRNN(units=50, activation="tanh", return_sequences=True))
        model.add(Dropout(0.2))

        # Add the third SimpleRNN layer and add dropout regularization
        model.add(SimpleRNN(units=50, activation="tanh", return_sequences=True))
        model.add(Dropout(0.2))

        # Add the fourth SimpleRNN layer and add dropout regularization
        model.add(SimpleRNN(units=50))
        model.add(Dropout(0.2))

        # Add the output layer
        model.add(Dense(units=1))

        # Compile the model
        model.compile(optimizer="adam", loss="mean_squared_error")

        return model

    def summary(self):
        return self.model.summary()

    def scale_data(self, X, y=None):
        X_scaled = self.scaler.fit_transform(X)
        if y is not None:
            y_scaled = self.scaler.fit_transform(y.reshape(-1, 1))
            return X_scaled, y_scaled
        return X_scaled

    def train(self, X_train, y_train, epochs=100, batch_size=32):
        X_train_scaled, y_train_scaled = self.scale_data(X_train, y_train)
        X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
        return self.model.fit(X_train_scaled, y_train_scaled, epochs=epochs, batch_size=batch_size)

    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
        predictions_scaled = self.model.predict(X_test_scaled)
        predictions = self.scaler.inverse_transform(predictions_scaled)
        return predictions
    def evaluate(self, X_test, y_test):
        X_test_scaled, y_test_scaled = self.scale_data(X_test, y_test, fit=False)
        X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
        loss = self.model.evaluate(X_test_scaled, y_test_scaled)
        return loss

    def save_model(self, file_path):
        self.model.save(file_path)
        np.save(file_path + '_scaler_X.npy', self.scaler_X)
        np.save(file_path + '_scaler_y.npy', self.scaler_y)

    def load_model(self, file_path):
        
        self.model = load_model(file_path)
        self.scaler_X = np.load(file_path + '_scaler_X.npy', allow_pickle=True).item()
        self.scaler_y = np.load(file_path + '_scaler_y.npy', allow_pickle=True).item()
    
def RNN(data, features = "Open"):
    # Defining training size to be 80% of the overall data.
    training_size = int(len(data)*0.80)

    #Defining the Length of data.
    data_len = len(data)

    # Defining Training and Testing Data.
    train=data[0:training_size]
    test=data[training_size:data_len]
    print("Training Size --> ", training_size)
    print("total length of data --> ", data_len)
    print("Train length --> ", len(train))
    print("Test length --> ", len(test))
    train = train.loc[:, [features]].values

    # Defining range for Scaling.
 
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train)
    end_len = len(train_scaled)
    X_train = []
    y_train = []
    timesteps = 40

    # Splitting the data into "X" and "Y" terms.
    for i in range(timesteps, end_len):
        X_train.append(train_scaled[i - timesteps:i, 0])
        y_train.append(train_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Reshaping the data as per as our need.
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    print("X_train --> ", X_train.shape)
    print("y_train shape --> ", y_train.shape)
    
    # Initialize a sequential model
    regressor = Sequential()  

    # Add the first SimpleRNN layer and add dropout regularization
    regressor.add(SimpleRNN(units=50, activation="tanh", return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))  

    # Add the second SimpleRNN layer and add dropout regularization
    regressor.add(SimpleRNN(units=50, activation="tanh", return_sequences=True))
    regressor.add(Dropout(0.2))  # Add dropout regularization

    # Add the third SimpleRNN layer and add dropout regularization
    regressor.add(SimpleRNN(units=50, activation="tanh", return_sequences=True))
    regressor.add(Dropout(0.2)) 

    # Add the fourth SimpleRNN layer and add dropout regularization
    regressor.add(SimpleRNN(units=50))
    regressor.add(Dropout(0.2)) 

    # Add the output layer
    regressor.add(Dense(units=1))
    
    regressor.compile(optimizer= "adam", loss = "mean_squared_error")
    
    epochs = 100 
    batch_size = 20
    
    regressor.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)
    
    real_price = test.loc[:, ["Open"]].values
    # print("Real Price Shape --> ", real_price.shape)
    
    # Concatenate the "Open" values from data and test datasets
    dataset_total = pd.concat((data["Open"], test["Open"]), axis=0)

    # Extract inputs for the test dataset with consideration of timesteps
    inputs = dataset_total[len(dataset_total) - len(test) - timesteps:].values.reshape(-1, 1)

    # Transform inputs using the previously fitted scaler
    inputs = scaler.transform(inputs)

    # Initialize an empty list to store test input sequences
    X_test = []

    # Iterate over the range to create input sequences
    for i in range(timesteps, 412):
        X_test.append(inputs[i - timesteps : i, 0])

    # Convert the list of sequences into a numpy array
    X_test = np.array(X_test)

    # Print the shape of the X_test array
    # print("X_test shape --> ", X_test.shape)
    
    # Reshape X_test to match the input shape required by the model
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Make predictions using the trained regressor model
    predict = regressor.predict(X_test)

    # Inverse transform the predictions to original scale
    predict = scaler.inverse_transform(predict)
    
    return predict, real_price