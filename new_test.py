from os import name as os_name
from os import getcwd, path, listdir
from PIL import Image
from numpy import asarray , load
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import os
from numpy import expand_dims
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
import cv2
import time
from pathlib import Path
from sklearn.preprocessing import Normalizer
from training import train_model

# device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(
  image_size=160, margin=0, keep_all=True, thresholds=[0.6, 0.7, 0.7], min_face_size=20, 
  factor=0.709, post_process=True 
)

print ("Model Loaded!")

VIDEO_PATH = "rtsp://admin:JGPHAN@192.168.0.104:554"
EMPLOYEES_NAMES = os.getcwd()

def image_resize(cv_img, k1 = 640, k2 = 480):
    x = cv_img.shape[0]
    y = cv_img.shape[1]
    if (x < y):
        x, y = y, x
    return cv2.resize(cv_img, (0, 0), fx = (k1/x), fy =k2/y)


def extract_face_from_file(filename, required_size=(160, 160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    results = mtcnn.detect(pixels,landmarks=True)
    # if (len(results)!=0):
    #     print ("sssssssss", results)
    boxes, probs , landmarks = results
    # boxes = boxes[0].tolist()
    print ("final boxessssss ::::  ", boxes)
    # x1, y1 = boxes[0] , boxes [1]
    # x1, y1 = int(x1), int(y1)
    # x2, y2 = boxes[2], boxes [3]
    # x1, y1 = int(x2), int(y2)
    # print("boxes::::",boxes,"probss:::::", probs, "Landmarks:::",landmarks)
    # print("box type::::", type(boxes),"probs type::::", type(probs),"land type::::", type(landmarks),)
    print("TOLIST=====", boxes[0].tolist())
    print("PIXELSsss===", pixels)
    print ("NP MESH GRID=======",np.meshgrid(boxes[[0, 2]], boxes[[1, 3]]))
    face = pixels[1]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    cv2.imshow("image",  image)
    # face_array = asarray(image)
    # return face_array


def extract_face_from_frame(pixels, required_size=(160, 160)):
    results = mtcnn.detect(pixels)
    if (len(results)==1):  
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        face_array = cv2.resize(face, required_size)
        
        return face_array   
    else:
        return []
    

def get_embedding(model, face_pixels):
	face_pixels = face_pixels.astype('float32')

	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	samples = expand_dims(face_pixels, axis=0)
	yhat = model.predict(samples)
	return yhat[0]


def initiate_known_embed(filename):
    frame = cv2.imread(filename)
    print ("this is frame : ", frame)
    face = extract_face_from_frame(frame)
    embed = get_embedding(resnet, face)
    in_encoder = Normalizer(norm='l2')
    norm_encod = in_encoder.transform([embed])
    return norm_encod


def getFaceEmbeddingsFromImage(face_img):
    embed = get_embedding(resnet, face_img)
    in_encoder = Normalizer(norm='l2')
    return in_encoder.transform([embed])

def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))    
    probability = np.linalg.norm(face_encodings - face_to_compare, axis=1)
    print(probability)
    return probability

def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.5):
    return (face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)

def get_results_with_detected_face(face, known_embed):
    return compare_faces(known_embed, getFaceEmbeddingsFromImage(face))[0]

def initiate_embeddings_from_file(filename):
    frame = cv2.imread(filename)
    face = extract_face_from_frame(frame)
    embed = get_embedding(resnet, face)
    in_encoder = Normalizer(norm='l2')
    norm_encod = in_encoder.transform([embed])
    return norm_encod

def initiate_embeddings_from_array(frame):
    face = extract_face_from_frame(frame)
    embed = get_embedding(resnet, face)
    in_encoder = Normalizer(norm='l2')
    norm_encod = in_encoder.transform([embed])
    return norm_encod


def setEmbeddingsInFile(personpath, calculate_encodings = []):
    if (calculate_encodings == []):
        known_embed = initiate_known_embed(personpath)
    else:
        known_embed = calculate_encodings
    embeddings = known_embed[0]
    
    first_split = personpath.split('/')


    second_split = first_split[len(first_split)-1].split('.')
    filename = path+second_split[0]+'.txt'

    with open(filename, 'w') as filehandle:
        for listitem in embeddings:
            filehandle.write('%s\n' % listitem)

    filehandle.close()

    return second_split[0]+'.txt'

def getEmbeddingsFromFile(path):
    embed = []
    with open(path, 'r') as filehandle:
        for line in filehandle:
            currentPlace = line[:-1]
            embed.append(float(currentPlace))
    filehandle.close()
    return [embed]


def getResults(frame, embedding_file_path):
    face = extract_face_from_frame(frame)
    if (len(face)==0):
        print("returning false")
        return False
    else:
        file_embedding = getEmbeddingsFromFile(embedding_file_path)
        frame_embedding = getFaceEmbeddingsFromImage(face)
        return (compare_faces(file_embedding, frame_embedding)[0])
    
def get_video_from_camera(vid_path):
    vid = cv2.VideoCapture(vid_path)
    ret = True
    while ret:
        ret, frame = vid.read()
        # frame = image_resize(frame)
        frame = face_match(frame,'data.pt')
        show_camera(frame)
        # cam_face = extract_face_from_frame(frame)
        # face_embeddeing = get_embedding (resnet, cam_face)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


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

def plot_boxes(frame):
    batch_boxes, cropped_images = detect_box(frame)
    if cropped_images is not None:
            for box, cropped in zip(batch_boxes, cropped_images):
                x, y, x2, y2 = [int(x) for x in box]
                resnet.classify = True
                img_embedding = getFaceEmbeddingsFromImage(extract_face_from_frame(frame))
                # prob = compare_faces(known_embedding, img_embedding)

## This fuction will take folder path and create and embedding file with names (as folders) and encodings of 
## persons inside those folders 
def load_dataset_from_folder(img_path):
    flag = None
    X, y = list(), list()
    if os.path.isfile( EMPLOYEES_NAMES +'/EncodingFile.npz') == True:
        data = load( EMPLOYEES_NAMES + '/EncodingFile.npz', allow_pickle=True)
        Employees = data['arr_0'] # List
        # Already exist
        flag = False
    else:
        Employees = np.array([])
        flag = True
  
    print("CURRENT EMPLOYEE DATABASE:" ,Employees )
    ids = []
    face_embeddings = []
    for subdir in listdir(img_path):
        if subdir in Employees:
            continue
        Employees = np.append(Employees , subdir)
        path = img_path + subdir + '/' # PATH = SUBFOLDER
        if not os.path.isdir(path): # skip any files that might be in the dir
            print("WARNING: FILES EXIST IN THE DATA DIRECTORY (ONLY FOLDERS ARE READ)!")
            print("SKIPPING FILE" , path , "...")
            continue
        break


def collate_fn(x):
    return x[0]

def face_match(img, data_path): # img_path= location of photo, data_path= location of data.pt 
    # getting embedding matrix of the given img
    # img = Image.open(img_path)
    face, prob = mtcnn(img, return_prob=True) # returns cropped face and probability
    
    if face is not None:
        face = face.cuda()
        emb = resnet(face.unsqueeze(0)).detach() # detach is to make required gradient false
    
        saved_data = torch.load('data.pt') # loading data.pt file
        embedding_list = saved_data[0] # getting embedding data
        name_list = saved_data[1] # getting list of names
        dist_list = [] # list of matched distances, minimum distance is used to identify the person
    
        for idx, emb_db in enumerate(embedding_list):
            dist = torch.dist(emb, emb_db).item()
            dist_list.append(dist)
        
        idx_min = dist_list.index(min(dist_list))
        plot_boxes(img)
        return img
        # return (name_list[idx_min], min(dist_list))
            


def show_camera(frame):
    cv2.imshow("Camera", frame)

if __name__ == "__main__":
    #setEmbeddingsInFile("./saved/")
    train_model('saved')
    # load_dataset_from_folder("./saved/")
    get_video_from_camera(VIDEO_PATH)
     
