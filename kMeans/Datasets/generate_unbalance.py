from PIL import Image
import numpy as np
import os


# unbalance.py is from http://cs.uef.fi/sipu/datasets/ (and in turn from Rezaei and Fränti)

data_vectors = []


with open('unbalance.txt') as f:
    for line in f.readlines():
        data_vectors.append(list(map(float, line.strip().split())))
        
np.save('unbalance_data_vectors.npy', data_vectors)
