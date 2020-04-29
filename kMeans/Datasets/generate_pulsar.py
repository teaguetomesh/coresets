from PIL import Image
import numpy as np
import os


# HTRU2.csv is from https://archive.ics.uci.edu/ml/datasets/HTRU2

data_vectors = []


with open('HTRU_2.csv') as f:
    for i, line in enumerate(f.readlines()):
        data_vectors.append(list(map(float, line.strip().split(','))))
        
np.save('pulsar_data_vectors.npy', data_vectors)
