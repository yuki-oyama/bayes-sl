import numpy as np

def f(a, b, sigma=np.zeros(a.shape)):
    return a + b * sigma
