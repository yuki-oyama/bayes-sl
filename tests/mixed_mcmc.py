import numpy as np
from polyagamma import random_polyagamma
import scipy.sparse as sp
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse import linalg as splinalg
from sklearn.preprocessing import scale
from scipy.special import beta
from tqdm import tqdm
from icecream import ic
from scipy.stats import invwishart

# MCMC

def next_paramRnd(paramRnd,)
