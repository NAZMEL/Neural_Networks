import sources.distributions as distributions
import sources.generate_y as hi_square_work
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import expon

# generate normal distributions 
def generate_normal_distributions(batch = 1000, size = 1000):
    x_values = []
    for _ in range(batch):
        x_values.append(distributions.generate_normal_distribution(size = size))
    
    return np.array(x_values)

# generate exponentials distributions
def generate_expon_distributions(batch=1000, size = 1000):
    x_values = []
    for _ in range(batch):
        x = np.linspace(expon.ppf(0.01), expon.ppf(0.99), size)
        rv = expon()
        x_values.append(rv.pdf(x))

    return np.array(x_values)

# mix two distributions by delimiter
def generate_mix_normal_and_expon(normal_distributions, expon_distributions, delimiter = 5):
    mix_distributions = []
    counter = 0
    for n in range(1, len(normal_distributions) + 1):
        if n % delimiter == 0:
            mix_distributions.append(expon_distributions[counter])
            counter += 1
        else:
            mix_distributions.append(normal_distributions[n-1])

    return np.asarray(mix_distributions)

# mix distribution by shuffle
def mix_distributions(arr1, arr2):
    mix_distr = []
    [mix_distr.append(item) for item in arr1.tolist()]
    [mix_distr.append(item) for item in arr2.tolist()]
    np.random.shuffle(mix_distr)

    return mix_distr

# get array with 0 and 1
# 1 - distribution is normal
# 0 - distribution is not normal
def get_y_train(distributions):
    return np.asarray([[dist] for dist in hi_square_work.hi_square_working(distributions)])
