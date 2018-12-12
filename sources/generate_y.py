from sources.config import drills, normal_distribution
from sources.config import hi_critical_23_005, hi_critical_11_005

distributions = drills
middle_output = 'This distribution for drill №{0} is {1}. hi_square: {2},  hi_critical: {3}'
hi_critical = hi_critical_23_005

# calculate hi_square between 2 distributions
def hi_square(array1, array2):
    return sum([ ((i - j) ** 2) / j for i, j in zip(array1, array2)])

def hi_square_working(distributions, normal_distribution = normal_distribution):
    # y_values for study neural network
    y_train = []

    for item in range(len(distributions)):
        
        hi = round(hi_square(distributions[item], normal_distribution), 2)

        #print(f'\nDrill with gas-saturated thickness values: {", ".join([str(x) for x in distributions[item]])}')

        if(hi < hi_critical):
            #print(middle_output.format(item + 1, 'NORMAL', hi, hi_critical))
            y_train.append(1)
        else:
            #print(middle_output.format(item + 1, 'NOT NORMAL', hi, hi_critical))
            y_train.append(0)

    return y_train



if __name__ == '__main__':
    print('-' * 120)

    y_train = hi_square_working(distributions, normal_distribution)

    print('-' * 120)
    print(f'Result array: {y_train}')