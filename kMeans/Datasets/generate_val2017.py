from PIL import Image
import image_vector
import numpy as np
import os

data_vectors = []
vector_to_label = {}


for i, f in enumerate(os.listdir('./val2017')):
    if '.jpg' in f:
        try:
            with Image.open('val2017/%s' % f) as im:
                vec = image_vector.Img2Vec().get_vec(im)
                data_vectors.append(vec)
                vector_to_label[vec.tobytes()] = '%s' % f
        except:
            pass
        
np.save('val2017_data_vectors.npy', data_vectors)
np.save('val2017_vector_to_label.npy', vector_to_label)
