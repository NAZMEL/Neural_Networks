'''
    Back propagation neural network.
'''
import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import SGD
import numpy as np
from sources.generator_distributions import mix_distributions, generate_normal_distributions, generate_expon_distributions, get_y_train
from sources import config
import time

X_values = config.drills
y_values = config.y_train
   
if __name__ == '__main__':

    model = Sequential()

    # lays of neurons
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy', 
        optimizer=SGD(lr = 0.1),
        metrics=['accuracy'])

    # training network
    t_start = time.perf_counter()
    model.fit(X_values, y_values, epochs=20, validation_split=0.2)
    t_stop = time.perf_counter()

    # testing on synthetic values
    normal_distr = generate_normal_distributions(1000, 24)
    expon_distr = generate_expon_distributions(450, 24)

    X_test = mix_distributions(normal_distr, expon_distr)
    X_test = np.asarray(X_test)
    y_test = get_y_train(X_test)

    _ , accuracy = model.evaluate(X_test, y_test)          # => loss, evaluate
    print(f"Success evaluate: {accuracy * 100}%\n")
    print(f"Time of fit: {round(t_stop - t_start, 3)}")