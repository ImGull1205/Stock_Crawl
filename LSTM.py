import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras import backend

def create_sequences(data, seq_length):
        xs = []
        ys = []
        for i in range(len(data) - seq_length):
            x = data[i:i+seq_length]
            y = data[i+seq_length][-1]  # Predict the 'Change' value
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

def LSTM(data, test_scale = 0.03):
    
    data['Change'].fillna(0, inplace=True)
    # Select features and target
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Change']
    target = 'Change'

    # Normalize the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[features])

    # Create sequences for LSTM
    

    seq_length = 3
    X, y = create_sequences(scaled_data, seq_length)

    
    # Split the data into training and testing sets
    train_size = int(len(X) * (1-test_scale))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build the LSTM Model
    model = Sequential()
    model.add(LSTM(units=150, return_sequences=True, input_shape=(seq_length, len(features))))  # Adjust LSTM units
    model.add(LSTM(units=150))  # Adjust LSTM units
    model.add(Dense(1))

    # Compile the model with a custom learning rate
    learning_rate = 0.001  # Adjust learning rate here
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Train the Model
    epochs = 150  # Adjust number of epochs here
    batch_size = 10  # Adjust batch size here
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

    # Step 5: Evaluate the Model
    loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}')

    # Step 6: Make Predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(np.concatenate((np.zeros((predictions.shape[0], len(features) - 1)), predictions), axis=1))[:, -1]

    # Inverse transform actual values for comparison
    y_test_inverse = scaler.inverse_transform(np.concatenate((np.zeros((y_test.shape[0], len(features) - 1)), y_test.reshape(-1, 1)), axis=1))[:, -1]
    
    return predictions, y_test_inverse, model

def predict_future (data, model):
    future_data = data[-seq_length:].copy()  # Assuming you want to predict 5 future time steps
    # Make sure to select the same features used during training
    future_features = future_data[features]

    # Scale the future features
    scaled_future_data = scaler.transform(future_features)

    # Generate sequences for the future data
    X_future, _ = create_sequences(scaled_future_data, seq_length)

    # Make predictions
    future_predictions = model.predict(X_future)

    # Inverse transform the predictions
    future_predictions = scaler.inverse_transform(np.concatenate((np.zeros((future_predictions.shape[0], len(features) - 1)), future_predictions), axis=1))[:, -1]
    
    return future_predictions
