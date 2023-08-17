import os
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy
import torch
import pickle
from pdb import set_trace as debug 


workers = 0 if os.name == 'nt' else 4
tdevice = torch.device('cuda' if torch.cuda.is_available else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(tdevice)
mtcnn = MTCNN(
  image_size=160, margin=0, keep_all=True, thresholds=[0.6, 0.7, 0.7], min_face_size=20, 
  factor=0.709, post_process=True , device=tdevice
)

def encode(img):
    res = resnet(torch.Tensor(img))
    # print (torch.Tensor(img).size())
    return res

def collate_fn(x):
    return x[0]

def get_encoding():
    print ("entered training")
    saved_pictures = "./saved/"
    dataset = datasets.ImageFolder(saved_pictures)
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)


    # pic_list = os.listdir(saved_pictures)
    # # print (pic_list)
    people_ids = []
    encoded__list = []


    for x , y in loader:
        x_aligned, prob = mtcnn(x, return_prob=True)
        # x_aligned, prob = x_aligned.cuda() , prob.cuda()

        if x_aligned is not None:
            # debug()
            # print('Face detected with probability: {}'.format(prob))
            encoded__list.append(x_aligned)
            people_ids.append(dataset.idx_to_class[y])
            # print("appending values:  ", encoded__list, people_ids)


    encoded__list = torch.stack(encoded__list).to(tdevice)
    encoded__list = encoded__list.squeeze()
    all_people_faces = resnet(encoded__list)

    # for file in (pic_list):
    #     img = cv2.imread(os.path.join(saved_pictures, file))
    #     people_ids.append(os.path.splitext(file)[0])
    #     print("img saved!!!!")
    #     print("people ids ", people_ids)
    #     cropped = mtcnn(img)
    #     if cropped is not None:
    #         encoded__list.append (encode(cropped)[0,:])
    print("done encoding")
    # for item in encoded__list:
    #     # cv2.imshow("pic", item)
    #     print("picture = ", item)
    # for key in people_ids:
    #     for val in encoded__list:
    #         all_people_faces[key] = val
    #         encoded__list.remove(val)
    #         break
    enc = open("EncodingFile.npz", 'wb')
    pickle.dump(all_people_faces, enc)
    enc.close()
    return all_people_faces, people_ids