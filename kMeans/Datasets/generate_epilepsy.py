from PIL import Image
import numpy as np
import os


# epilepsy.csv is from https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition

data_vectors = []


with open('epilepsy.csv') as f:
    for i, line in enumerate(f.readlines()):
        if i == 0:
            continue
        data_vectors.append(list(map(float, line.strip().split(',')[1:])))
        
np.save('epilepsy_data_vectors.npy', data_vectors)
