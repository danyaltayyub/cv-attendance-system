from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
from types import MethodType
import cv2

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion

dataset=datasets.ImageFolder('saved') # photos folder path 
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # accessing names of peoples from folder names

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

loader = DataLoader(dataset, collate_fn=collate_fn)

face_list = [] # list of cropped faces from photos folder
name_list = [] # list of names corrospoing to cropped photos
embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

for img, idx in loader:
    face, prob = mtcnn(img, return_prob=True) 
    if face is not None and prob>0.90: # if face detected and porbability > 90%
        emb = resnet(face.unsqueeze(0)).detach() # passing cropped face into resnet model to get embedding matrix
        embedding_list.append(emb) # resulten embedding matrix is stored in a list
        name_list.append(idx_to_class[idx]) # names are stored in a list


data = [embedding_list, name_list]
torch.save(data, 'data.pt') # saving data.pt file

def face_match(img, data_path): # img_path= location of photo, data_path= location of data.pt 
    # getting embedding matrix of the given img
    # img = Image.open(img_path)
    face, prob = mtcnn(img, return_prob=True) # returns cropped face and probability
    emb = resnet(face.unsqueeze(0)).detach() # detech is to make required gradient false
    
    saved_data = torch.load('data.pt') # loading data.pt file
    embedding_list = saved_data[0] # getting embedding data
    name_list = saved_data[1] # getting list of names
    dist_list = [] # list of matched distances, minimum distance is used to identify the person
    
    for idx, emb_db in enumerate(embedding_list):
        dist = torch.dist(emb, emb_db).item()
        dist_list.append(dist)
        
    idx_min = dist_list.index(min(dist_list))
    return (name_list[idx_min], min(dist_list))



cap = cv2.VideoCapture("rtsp://admin:JGPHAN@192.168.0.104:554")
while (1):
    ret, frame = cap.read()
    if (ret == True) :
        # cv2.imshow('frame', frame)
        boxes , faces = mtcnn.detect_box(frame)

        if faces is not None:
            for box , cropped in zip (boxes , faces):
                x, y, x2, y2 = [int(x) for x in box]
                resnet.classify = True
                result = face_match(cropped , 'data.pt')
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                  frame, result [0], (x + 5, y + 10), 
                   cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)



    else :
        print("No video")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# result = face_match('Danyal.jpg', 'data.pt')

# print('Face matched with: ',result[0], 'With distance: ',result[1])