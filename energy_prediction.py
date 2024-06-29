#%% Importing the libraries

import pandas as pd
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split

#%% Importing the dataset

df = pd.read_excel('Folds5x2_pp.xlsx')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#%% Splitting the data into train and testset

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)

#%% Creating the Neural Network

ann = tf.keras.Sequential()
ann.add(keras.layers.Dense(units=6, activation='relu'))
ann.add(keras.layers.Dense(units=6, activation='relu'))
ann.add(keras.layers.Dense(units=1))

#%% Training the ann

ann.compile(optimizer='adam', loss='mean_squared_error')
ann.fit(X_train, y_train, batch_size=32, epochs=100)

#%% Prediction

y_pred = ann.predict(X_test)