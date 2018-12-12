'''
    Counter propagation neural network.
'''
import numpy as np
from sources.generator_distributions import mix_distributions, generate_normal_distributions, generate_expon_distributions, get_y_train
from sources import config
from math import sqrt
import time

X_values = config.drills
y_values = config.y_train

def sum_squares(arr):
    return sum([x ** 2 for x in arr])

# function normalization for vectors X
def normalization(arr_x, arr_y):
    sum_y_squares = sum_squares(arr_y)
    result = []

    for row_x in arr_x:
        middle_result = []
        sum_x_squares = sum_squares(row_x)
        for x in row_x:
            middle_result.append(round(x / sqrt(sum_x_squares + sum_y_squares), 2))
        
        result.append(middle_result)
    return result
 
class Counter_Propagation:
    def __init__(self, X_values, y_values, kohonen_neurons = 2, grossberg_neurons = 1, len_x_vector = 4):
        self.X_values = X_values
        self.y_values = y_values

        self.kohonen_weights = self.generate_weights(kohonen_neurons, len_x_vector)
        self.grossberg_weights = self.generate_weights(grossberg_neurons, len_x_vector)

    # generate weights for each part of network
    def generate_weights(self, num_neurons = 1, length=4):
        result = np.asarray(np.random.rand(num_neurons, length))
        
        if len(result) == 1:
            return result[0]
        return result

    # for shorter Evklide's way       
    def calculate_evklid_way(self, w_vector, x_vector):
        return sum([ ((w-x) ** 2) for w, x in zip(w_vector, x_vector)])

    # calculate net for Grossberg lay
    def sum_activation(self, k_vector, w_vector):
        return sum([ k*w for k, w in zip(k_vector, w_vector)])

    # update vector kohonen weights
    def update_kohonen_weights(self, x_vector, w_vector, learning_rate = 0.7):
        weights = []

        for x, w in zip(x_vector, w_vector):
            w_new = w + learning_rate * (x - w)
            weights.append(w_new)

        return np.asarray(weights)

    # update weights for grossberg lay
    def update_grossberg_weights(self, y_value, w_value, learning_rate = 0.1, k = 1):
        w_new = w_value + learning_rate * (y_value - w_value) * k
        #print(f'grossberg update: {w_value}, {w_new}')
        return w_new

    # counter of training success
    def good_count(self, y_value, out_network):
        if y_value == out_network:
            return 1
        return 0
        
    # training counter propagation neural network
    def fit(self, lr_kohonen = 0.7, lr_grossberg = 0.1, epochs = 10):
        for epoch in range(epochs):
            if epoch % 5 == 0 and lr_kohonen > 0.1 and lr_grossberg > 0.01:
                lr_kohonen-= 0.05
                lr_grossberg -= 0.005
            
            good_counter = 0
            for x_vector, y_value in zip(self.X_values, self.y_values):
                kohonen_neurons = []
                
                for w_iter in range(len(self.kohonen_weights)):
                    kohonen_neurons.append(self.calculate_evklid_way(x_vector, self.kohonen_weights[w_iter]))

                neuron_min = min(kohonen_neurons)                                                               
                index = kohonen_neurons.index(neuron_min)    

                for i in range(len(kohonen_neurons)): 
                    if i == index:
                        kohonen_neurons[i] = 1
                    else:
                        kohonen_neurons[i] = 0 

                self.kohonen_weights[index] = self.update_kohonen_weights(x_vector, self.kohonen_weights[index], learning_rate= lr_kohonen)

                # grossberg neurons
                self.grossberg_weights[index] = self.update_grossberg_weights(y_value, self.grossberg_weights[index], learning_rate=lr_grossberg)   
                grossberg_neuron_out = int(round(self.sum_activation(kohonen_neurons, self.grossberg_weights)))
                good_counter += self.good_count(y_value, grossberg_neuron_out)

                print(f'{y_value} : {grossberg_neuron_out}')
                
            print(f'Success training {epoch+1} epoch: {int(good_counter/len(self.y_values) * 100)}%')
    

    # work network on test values
    def evaluate(self, X_values, y_values):
        self.X_values = X_values
        self.y_values = y_values

        good_counter= 0
        for x_vector, y_value in zip(self.X_values, self.y_values):
            kohonen_neurons = []
                
            for w_iter in range(len(self.kohonen_weights)):
                kohonen_neurons.append(self.calculate_evklid_way(x_vector, self.kohonen_weights[w_iter]))

            neuron_min = min(kohonen_neurons)                                                               
            index = kohonen_neurons.index(neuron_min)    

            for i in range(len(kohonen_neurons)):    
                if i == index:
                    kohonen_neurons[i] = 1
                else:
                    kohonen_neurons[i] = 0 
              
            grossberg_neuron_out = int(round(self.sum_activation(kohonen_neurons, self.grossberg_weights)))
            good_counter += self.good_count(y_value, grossberg_neuron_out)

        print(f'Success evaluate: {int(good_counter/len(self.y_values) * 100)}%')

# work neural network
if __name__ == '__main__':
    
    net = Counter_Propagation(X_values, y_values, kohonen_neurons=2, grossberg_neurons=1, len_x_vector=len(X_values[0]))

    t_start = time.perf_counter()
    net.fit(lr_kohonen=0.7, lr_grossberg=0.1, epochs=2)
    t_stop = time.perf_counter()

    print(f"Time of fit: {round(t_stop - t_start, 3)}")
    
    # testing on synthetic values
    normal_distr = generate_normal_distributions(1000, 24)
    expon_distr = generate_expon_distributions(450, 24)

    X_test = mix_distributions(normal_distr, expon_distr)
    y_test = get_y_train(X_test) 

    print("Evaluate")
    net.evaluate(X_test, y_test)

    