import numpy as np
'''
Takes albedo and/or diameter data from WISE and generates a sample of arbitrary
size that imitates the WISE data
'''

import pandas as pd
from auxiliary_functions import imitate_sample



file='neowise_jupiter_trojans.csv'
n_bins=20 # number of bins for the available data
N = 10000 # number of output objects

H, D, aV, aIR = (pd.read_csv(file, usecols=[3,11,13,15])).T.to_numpy()


variable=aV

# -0.999 is a flag for not having albedo
index = np.argwhere(variable==-0.999)
variable = np.delete(variable, index)


synthetic_sample=imitate_sample(variable, n_bins, N)

