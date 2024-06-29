#%% Importing the libraries

import pandas as pd
import tensorflow.keras as keras
import tensorflow.keras.layers as layers 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#%% Importing the dataset

df = pd.read_excel('Folds5x2_pp.xlsx')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#%% Splitting the data into train and testset

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)

#%% Creating the Neural Network

ann = keras.Sequential()
ann.add(layers.Dense(units=6, activation='relu'))
ann.add(layers.Dense(units=6, activation='relu'))
ann.add(layers.Dense(units=1))

#%% Training the ann

ann.compile(optimizer='adam', loss='mean_squared_error')
ann.fit(X_train, y_train, batch_size=32, epochs=100)

#%% Prediction

y_pred = ann.predict(X_test)
y_pred = y_pred.flatten()

#%% Evaluating the model performance

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)