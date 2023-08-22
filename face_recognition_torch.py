import cv2
import torch
import pandas as pd
import numpy as np
from os import listdir
from facenet_pytorch import InceptionResnetV1, MTCNN
from tqdm import tqdm
from types import MethodType
from sklearn.preprocessing import Normalizer

class FaceRecognition:
    def __init__(self):
        # Constants
        self.BATCH_SIZE = 8
        self.FRAME_SKIP = 5

        # Face detection and recognition models
        self.resnet = None  # InceptionResnetV1 for face recognition
        self.mtcnn = None   # MTCNN for face detection
        self.image_size = 160
        self.margin = 0
        self.keep_all = True
        self.thresholds = [0.6, 0.7, 0.7]
        self.min_face_size = 40
        self.factor = 0.709
        self.post_process = True

        # Load required modules
        self.__load_modules__()

    def detect_box(self, img, save_path = None):
        """
        Detect faces in an image and extract bounding boxes.

        Args:
            img (numpy.ndarray): Input image.
            save_path (str, optional): Path to save the processed image. Defaults to None.

        Returns:
            tuple: Detected bounding boxes and extracted faces.
        """
        # Detect faces
        batch_boxes, batch_probs, batch_points = self.mtcnn.detect(img, landmarks=True)
        # Select faces
        if not self.keep_all:
            batch_boxes, batch_probs, batch_points = self.select_boxes(
                batch_boxes, batch_probs, batch_points, img, method=self.selection_method
            )
        # Extract faces
        faces = self.mtcnn.extract(img, batch_boxes, save_path)
        
        if faces is not None:
            return batch_boxes, faces

    def __load_modules__(self):
        """
        Load MTCNN and InceptionResnetV1 models for face detection and recognition.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(
            image_size=self.image_size,
            margin=self.margin,
            keep_all=self.keep_all,
            thresholds=self.thresholds,
            min_face_size=self.min_face_size,
            factor=self.factor,
            post_process=self.post_process,
            device=device
        )

        # Load InceptionResnetV1 for face recognition
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

        # Attach custom method to MTCNN instance
        self.mtcnn.detect_box = MethodType(self.detect_box, self.mtcnn)


    def img_resize(self, cv_img, k1 = 640, k2 = 480):
        x = cv_img.shape[0]
        y = cv_img.shape[1]
        if (x < y):
            x, y = y, x
        return cv2.resize(cv_img, (0, 0), fx = (k1/x), fy =k2/y)
    
    def extract_face_from_file(self, img_path):
        img = cv2.imread(img_path)
        faces = self.extract_face_from_frame(img)
        if faces is not None:
            return faces
        else:
            return []
    
    def extract_face_from_frame(self, frame):
        frame = self.img_resize(frame)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _ , faces = self.mtcnn.detect_box(rgb_image)
        return faces
    
    def get_embeddings(self, face):
        face = face.cuda()
        embedding = self.resnet (face.unsqueeze(0))
        return embedding.cpu().detach().numpy()
    
    def initiate_known_embed(self, filename):
        face = self.extract_face_from_file(filename)
        embed = self.get_embeddings(face)
        in_encoder = Normalizer(norm='l2')
        norm_encod = in_encoder.transform([embed])
        return norm_encod
    
    def initiate_embed_from_frame(self, frame):
        face = self.extract_face_from_frame(frame)
        embed = self.get_embeddings(face)
        in_encoder = Normalizer(norm='l2')
        norm_encod = in_encoder.transform([embed])
        return norm_encod
    

    def train(self, database):
        name_list = [] # list of names corrospoing to cropped photos
        embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

        for ls in listdir(database):
            subdir = database + '/' + ls
            # temp_list = []
        
            for subls in listdir(subdir):
                emb = self.initiate_known_embed(subdir + '/' + subls)
                embedding_list.append(emb)
                name_list.append(ls)

        
        data = [embedding_list,name_list]
        torch.save(data, 'data2.pt') # saving data.pt file
        print("\n✔ Training done.")
        print(f"✔ Total Persons in our Dataset: {len(name_list)}")
        for p in name_list:
            print(f"\t\t✅{p}")






# Example usage
if __name__ == "__main__":
    # Initialize face recognition
    face_recognition = FaceRecognition()

    # Load input image using OpenCV
    image_path = "oa.jpg"
    img = cv2.imread(image_path)

    encoded_test = face_recognition.initiate_known_embed(image_path)
    # Perform face detection and recognition
    boxes, faces = face_recognition.detect_box(img)

    # Display detected bounding boxes and extracted faces
    for box, face in zip(boxes, faces):
        x1, y1, x2, y2 = [int(a) for a in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.imwrite("output.jpg", img)
    cv2.imshow("Detected Faces", img)

