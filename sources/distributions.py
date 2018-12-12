'''
    Generate normal distribution which will be using for generate test data.
    This data will be using for comparison with data in table by Pearson's criterian. 
'''
from scipy.stats import truncnorm

# return normal distribution with 24 values
def normal_distribution():
    arr =  [2, 3, 4, 5, 4, 3, 2, 1]
    
    normal_dist = []

    for i in range(1, len(arr)):
        for _ in range(arr[i-1]):
            normal_dist.append(i)

    return normal_dist

# from scipy
def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def generate_normal_distribution(size = 24):
    X = get_truncated_normal(mean=15, sd=1, low=1, upp=30)
    return X.rvs(size)

if __name__ == '__main__':

    distribution = generate_normal_distribution(size = 12)
    print(distribution)