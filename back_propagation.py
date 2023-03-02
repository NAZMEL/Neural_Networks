'''
    Back propagation neural network.
'''
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

feature_set_x, labels_y = datasets.make_moons(1000, noise=0.2)
X_values, X_test, y_values, y_test = train_test_split(feature_set_x, labels_y, test_size=0.33, random_state=42)
model = Sequential()
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(
    loss='binary_crossentropy', 
    optimizer=SGD(lr = 0.1),
    metrics=['accuracy'])
t_start = time.perf_counter()
model.fit(X_values, y_values, epochs=200, validation_split=0.2)
t_stop = time.perf_counter()
_ , accuracy = model.evaluate(X_test, y_test)          
print(f"Success evaluate: {accuracy * 100}%\n")
print(f"Time of fit: {round(t_stop - t_start, 3)}")