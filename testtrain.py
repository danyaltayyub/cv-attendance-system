from facenet_pytorch import MTCNN, InceptionResnetV1
from os import listdir
import cv2
from types import MethodType
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2',classify=True).eval().to(device)


def encode_from_cropped(cropim):
    cropim = cropim.cuda()
    embed = resnet (cropim.unsqueeze(0))
    return embed




def detect_box(self, img, save_path=None):
    # Detect faces
    batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)
    # Select faces
    if not self.keep_all:
        batch_boxes, batch_probs, batch_points = self.select_boxes(
            batch_boxes, batch_probs, batch_points, img, method=self.selection_method
        )
    # Extract faces
    faces = self.extract(img, batch_boxes, save_path)
    return batch_boxes, faces

mtcnn.detect_box = MethodType(detect_box, mtcnn)

def collate_fn(x):
        return x[0]

def train_model(path):

    face_list = [] # list of cropped faces from photos folder
    name_list = [] # list of names corrospoing to cropped photos
    embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet


    for ls in listdir(path):
        subdir = path + '/' + ls
        temp_list = []
        
        for subls in listdir(subdir):
            img = Image.open(subdir + '/' + subls)
            face , prob = mtcnn(img, return_prob=True)

            if face is not None and prob>0.9:
                face_list.append(face)
                embed = encode_from_cropped(face)
                embedding_list.append(embed)
                name_list.append(ls)
                print ("probability of ", ls , " is ", prob)

    

    data = [embedding_list, name_list]
    torch.save(data, 'data.pt') # saving data.pt file



    print ("TEESSSTTIINNNGGGG+++++++==========")

    timg = cv2.imread("test.jpg")
    
    boxes, tfaces = mtcnn.detect_box(timg)
    if tfaces is not None:
        # x, y, x2, y2 = [int(x) for x in box]
        tembed = encode_from_cropped(tfaces)


        dist_list = [] # list of matched distances, minimum distance is used to identify the person
        
        for idx, emb_db in enumerate(embedding_list):
            dist = (tembed - emb_db).norm().item()
            print(name_list[idx], dist)
            dist_list.append(dist)


    print("Training done")
    return embedding_list, name_list

face_list , name_list = train_model('saved')
# print ("ITEMS in FACE LISTTT======", len(face_list))
# print ("NAME LISTTT======", name_list)