# Neural_Networks
The algorithm was designed for studying and testing both Neural Networks: Counter Propagation and Back Propagation.

## Main excercise
To teach the neural network to define data which ones belong to the normal distribution. And checking learned results on the data that is generated synthetically.

## Main functions for Counter Propagation
+ `generate_weights()` - forming an array of weights.
+ `fit()` - the main studying function.
+ `evaluate()` - work network on the test values.

The first value of learning rate is 0.7 for Kohonen's map. And it is 0.1 for Grossberg's map. In the process of learning values of these coefficients decrease to the treshold level.


## Back Propagation
Libraries [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) were used for implementation this algorithm. There was made 2 layouts of neurons. The training rate is 0.1 for the neural network. Some synthetic data is making for test. And they get to inputs of the neural network.
