from PIL import Image
import numpy as np
import os


# yeast.txt is from http://cs.joensuu.fi/sipu/datasets/ (and in turn from UCI)

data_vectors = []


with open('yeast.txt') as f:
    for line in f.readlines():
        data_vectors.append(list(map(float, line.strip().split())))
        
np.save('yeast_data_vectors.npy', data_vectors)
