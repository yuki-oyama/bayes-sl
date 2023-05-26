import numpy as np

def f(a, b, sigma=np.zeros(a.shape)):
    return a + b * sigma


a = np.arange(10)
a
a[0:3]
a[3:6]
a[6:9]
