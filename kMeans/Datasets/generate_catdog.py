from PIL import Image
import image_vector
import numpy as np


N = 10
data_vectors = []
vector_to_label = {}


for i in range(1, N + 1):
    try:
        with Image.open('Catdog/Cat/%s.jpg' % i) as im:
            vec = image_vector.Img2Vec().get_vec(im)
            data_vectors.append(vec)
            vector_to_label[vec.tobytes()] = 'Cat/%s.jpg' % i
    except:
        pass
        
    try:
        with Image.open('Catdog/Dog/%s.jpg' % i) as im:
            vec = image_vector.Img2Vec().get_vec(im)
            data_vectors.append(vec)
            vector_to_label[vec.tobytes()] = 'Dog/%s.jpg' % i
    except:
        pass

np.save('catdog_data_vectors.npy', data_vectors)
np.save('catdog_vector_to_label.npy', vector_to_label)
