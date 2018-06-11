import numpy as np

# Learning rate [.00001, 1)
rand_range = -5 * np.random.rand(10) - 1
lrs = 10**rand_range
with open('lr.csv', 'w') as f:
    for i in range(len(lrs)):
        print(lrs[i])
        print('{:.5f}'.format(lrs[i]), sep=',', file=f)
