from facenet_pytorch import MTCNN, InceptionResnetV1
from os import listdir
import cv2
import numpy as np
from tqdm import tqdm 
import pandas as pd
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import os

device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def encode_from_cropped(cropim):
    resnet.classify = True
    cropim = cropim.cuda()
    embed = resnet (cropim.unsqueeze(0))
    return embed.cpu().detach().numpy()[0]

def get_processed_img(img):
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb_image

def collate_fn(x):
        return x[0]

def train_model(path, set_face_detection_confidence=0.97):
    face_list = []
    name_list = []
    embedding_list = []
    face_prob_each_person = []

    for ls in tqdm(os.listdir(path), desc='Model Training'):
        subdir = os.path.join(path, ls)
        temp_list = []
        face_prob = []

        for subls in os.listdir(subdir):
            # img = Image.open(os.path.join(subdir, subls))
            img = get_processed_img(cv2.imread(subdir + '/' + subls))
            face, prob = mtcnn(img, return_prob=True)
            face = face.to(device)

            face_prob.append((ls, subls, prob))

            if face is not None and prob >= set_face_detection_confidence:
                face_list.append(face) 
                emb = encode_from_cropped(face) # passing cropped face into resnet model to get embedding matrix
                embedding_list.append(emb)



        name_list.append(ls)
        face_prob_each_person.append(pd.DataFrame(face_prob, columns=['Person', 'Images', 'Face Probability']))

    print("\n✔ Training done.")
    print(f"✔ Total Persons in our Dataset: {len(name_list)}")
    for p in name_list:
      print(f"\t\t✅{p}")

    df_face = pd.concat(face_prob_each_person)
    df_face['Person'] = df_face["Person"].where(~df_face["Person"].duplicated(), "")
    file_ = 'face-detection-info-using-mtcnn.csv'
    df_face.to_csv(file_, index=False)
    print(f"✔ Save file ({file_}) where you can see all person face probablity.")

    data = [embedding_list, name_list]
    torch.save(data, 'data.pt') # saving data.pt file
    return embedding_list, name_list

embedding_list, name_list = train_model('saved')