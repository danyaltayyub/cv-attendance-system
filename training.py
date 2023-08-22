from facenet_pytorch import MTCNN, InceptionResnetV1
from os import listdir
import cv2
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image

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

def train_model(path):
    # dataset=datasets.ImageFolder(path) # photos folder path 
    # idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # accessing names of peoples from folder names


    # loader = DataLoader(dataset, collate_fn=collate_fn)

    face_list = [] # list of cropped faces from photos folder
    name_list = [] # list of names corrospoing to cropped photos
    embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

    # for img, idx in loader:
    #     face, prob = mtcnn(img, return_prob=True)
    #     # 
    #     if face is not None and prob>0.90: # if face detected and porbability > 90%
    #         face = face.cuda()
    #         face_list.append(face) 
    #         emb = resnet(face.unsqueeze(0)) # passing cropped face into resnet model to get embedding matrix
    #         embedding_list.append(emb.detach()) # resulten embedding matrix is stored in a list
    #         name_list.append(idx_to_class[idx]) # names are stored in a list


    for ls in listdir(path):
        subdir = path + '/' + ls
        # temp_list = []
        
        for subls in listdir(subdir):
            img = get_processed_img(cv2.imread(subdir + '/' + subls))
            face , prob = mtcnn(img, return_prob=True)

            if face is not None and prob>0.99:
                face_list.append(face) 
                emb = encode_from_cropped(face) # passing cropped face into resnet model to get embedding matrix
                # temp_list.append(emb.detach()) # resulten embedding matrix is stored in a list
                embedding_list.append(emb)
                name_list.append(ls)

                
        # temp_stack = torch.stack(temp_list)        
        # embedding_list.append (torch.mean(temp_stack, dim=0))
        # name_list.append (ls)

                # print("images =====  ", img)


    data = [embedding_list, name_list]
    torch.save(data, 'data.pt') # saving data.pt file


    print("Training done")
    print (embedding_list)
    return embedding_list, name_list

# face_list , name_list = train_model('saved')
# print ("ITEMS in FACE LISTTT======", len(face_list))
# print ("NAME LISTTT======", name_list)