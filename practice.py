import torch
import random
import cv2
import numpy as np
from training import get_processed_img, encode_from_cropped,mtcnn

from sklearn.metrics.pairwise import cosine_similarity

img = get_processed_img(cv2.imread("test11.jpg"))
face , prob = mtcnn(img, return_prob=True)
# Generate a list of random numbers within the range -1 to 1
random_numbers = np.array(encode_from_cropped(face)).reshape(1, -1)



data = torch.load('data.pt') # loading data.pt file
names = [i for i in data[1]]
emb = [i for i in data[0]]

for n,e in zip(names, emb):

    similarity = cosine_similarity(e.reshape(1, -1), random_numbers)
    print(f"Cosine Similarity with {n}:", similarity[0][0])

# 03000665414