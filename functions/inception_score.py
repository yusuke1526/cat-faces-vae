import os, glob
import glob
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from keras.datasets import mnist
from PIL import Image as pil_image

model = InceptionV3() # Load a model and its weights

def resizer(x):
    x_list = []
    for i in range(x.shape[0]):
        img = image.array_to_img(x[i, :, :, :].reshape(digit_size, digit_size, -1))
        img = img.resize(size=(299, 299), resample=pil_image.LANCZOS)
        x_list.append(image.img_to_array(img))
    return np.array(x_list)

def inception_score(x, batch_size=32):
    r = None
    n_batch = (x.shape[0]+batch_size-1) // batch_size
    for j in range(n_batch):
        x_batch = resizer(x[j*batch_size:(j+1)*batch_size, :, :, :])
        r_batch = model.predict(preprocess_input(x_batch)) # r has the probabilities for all classes
        r = r_batch if r is None else np.concatenate([r, r_batch], axis=0)
    p_y = np.mean(r, axis=0) # p(y)
    e = r*np.log(r/p_y) # p(y|x)log(P(y|x)/P(y))
    e = np.sum(e, axis=1) # KL(x) = Σ_y p(y|x)log(P(y|x)/P(y))
    e = np.mean(e, axis=0)
    return np.exp(e) # Inception score