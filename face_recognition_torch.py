import cv2
import torch
import pandas as pd
import csv
import numpy as np
from os import listdir
from facenet_pytorch import InceptionResnetV1, MTCNN
from tqdm import tqdm
from types import MethodType
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_distances

class FaceRecognition:
    def __init__(self):
        # Constants
        self.BATCH_SIZE = 8
        self.FRAME_SKIP = 10
        self.dist_thres = 0.5
        self.VIDEO_PATH = "rtsp://admin:JGPHAN@192.168.0.104:554"

        # Initialize attendance file
        self.file = self.initiate_att_file()

        # Face detection and recognition models
        self.resnet = None  # InceptionResnetV1 for face recognition
        self.mtcnn = None   # MTCNN for face detection
        self.image_size = 200
        self.margin = 0
        self.keep_all = True
        self.thresholds = [0.6, 0.7, 0.7]
        self.min_face_size = 30
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
        return batch_boxes, faces

    def __load_modules__(self):
        """
        Load MTCNN and InceptionResnetV1 models for face detection and recognition.
        """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
        # self.mtcnn.detect_box = MethodType(self.detect_box, self.mtcnn)

    def initiate_att_file():
        header = ['Name', 'ID', 'Date', 'Time in']
        f = open('att.csv', "w+")
        writer = csv.writer(f)
        writer.writerow(header)
        return f


    def img_resize(self, cv_img, k1 = 640, k2 = 480):
        x = cv_img.shape[0]
        y = cv_img.shape[1]
        if (x < y):
            x, y = y, x
        return cv2.resize(cv_img, (0, 0), fx = (k1/x), fy =k2/y)
    
    def extract_face_from_file(self, img_path):
        img = cv2.imread(img_path)
        boxes ,faces = self.extract_face_from_frame(img)
        return boxes , faces
    
    def extract_face_from_frame(self, frame):
        faces = []
        frame = self.img_resize(frame)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes , faces = self.detect_box(rgb_image)
        return boxes , faces
    
    def get_embeddings_from_face(self, face_tensor):
        self.resnet.classify = True
        face_tensor = face_tensor.cuda()
        embedding = self.resnet(face_tensor.unsqueeze(0))
        return embedding.cpu().detach().numpy()
    
    def initiate_known_embed(self, filename):
        norm_encod = []
        _,face = self.extract_face_from_file(filename)
        if face is not None:
            for f in face:
                embed = self.get_embeddings_from_face(f)
                embed = self.get_embeddings_from_face(f)
                norm_encod.append(self.encode_embedding(embed))
        
        return norm_encod
    
    def initiate_embed_from_frame(self, frame):
        norm_encod = []
        face = self.extract_face_from_frame(frame)
        if face is not None:
            for f in face:
                embed = self.get_embeddings_from_face(f)
                norm_encod.append(self.encode_embedding(embed))
        
        return norm_encod
    
    def encode_embedding (self , img_embed):
        in_encoder = Normalizer(norm = 'l2')
        encoded = in_encoder.transform(img_embed)
        return encoded
    

    def calc_face_dist(self, face_emb1 , face_emb2):
        dist = cosine_distances(face_emb1[0].reshape(1,-1) , face_emb2[0].reshape(1,-1))
        return dist[0][0]
    
    def get_data_from_emb_file(self, file_path):
        data = torch.load(file_path)
        embedding_list = data[0]
        name_list = data [1]

        return embedding_list , name_list
    
    def draw_box (self, img, box, name, dist):
        x1, y1, x2, y2 = [int(x) for x in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 70), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(img, dist, (x1 + 30, y2 - 30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

        return img

    def label_names_in_frame (self, vid_frame , emb_list , name_list):
        boxes , faces = self.detect_box(vid_frame)
        min_dist = ""

        if faces is not None:
            for box, cropped in zip(boxes, faces):
                img_embedding = self.get_embeddings_from_face(cropped)
                encoding = [self.encode_embedding(img_embedding)]

                dist_list = []
                for idx, emb_db in enumerate(emb_list):
                    dist = self.calc_face_dist(emb_db, encoding)
                    print(name_list[idx], dist)
                    dist_list.append(dist)
                print("-------------")

                idx_min = dist_list.index(min(dist_list))


                if dist_list[idx_min] <= self.dist_thres:
                    min_key = name_list[idx_min]
                    min_dist = str(round(dist_list[idx_min],2))
                    
                elif dist_list[idx_min] > self.dist_thres:
                    min_key = 'Undetected'

                vid_frame = self.draw_box (vid_frame , box , min_key , min_dist)

        return vid_frame
    
    

    def train_data(self, database):
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

    def detect_from_video(self , video_path):
        vid = cv2.VideoCapture(video_path)
        embeddings , names = self.get_data_from_emb_file("data2.pt")
        i = 1
        while vid.grab():
            try:

                _, v_frame = vid.read()
                if (i == self.FRAME_SKIP):
                    i = 1
                    # v_frame = self.img_resize(v_frame)

                    if (v_frame is not None):
                        v_frame = self.label_names_in_frame(v_frame, embeddings, names)
                        cv2.imshow("output", v_frame)
                        
                        
                
                i = i+1
                
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    break
            except Exception as e:
                # print("image is", v_frame)
                print(str(e))
                print("\n\n\n\n Reinitializing...")
                vid = cv2.VideoCapture(video_path)

    def get_video(self, vid_path, SKIP_FRAME = 30):
        vid = cv2.VideoCapture(vid_path)
        i = 1
        while vid.grab():
            try:
                _, v_frame = vid.read()

                if (i != SKIP_FRAME):
                    i = i+1
                else:            
                    # v_frame = self.img_resize(v_frame)
                    i=1
                    if (v_frame is not None):
                        cv2.imshow("output", v_frame)
                
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    break
            except Exception as e:
                # print("image is", v_frame)
                print(str(e))
                print("\n\n\n\n Reinitializing...")
                vid = cv2.VideoCapture(vid_path)

    
    def detect_picture(self, ref_img_path, test_img_path):
        ref_emb = self.initiate_known_embed(ref_img_path)
        test_emb = self.initiate_known_embed(test_img_path)
        dist = self.calc_face_dist(ref_emb , test_emb)

        print ("distance === == =", dist)


# Example usage
if __name__ == "__main__":
    # Initialize face recognition
    face_recognition = FaceRecognition()
    # face_recognition.detect_picture("/home/transdata/cv-attendance-system/saved/Javaria Kashaf/IMG_20230823_201126.jpg" , "test3.jpg")
    # face_recognition.train_data("./saved/")
    face_recognition.detect_from_video(face_recognition.VIDEO_PATH)
    # face_recognition.get_video(face_recognition.VIDEO_PATH)

    # # Load input image using OpenCV
    # image_path = "test2.jpg"
    # im2_path = "/home/transdata/cv-attendance-system/saved/Shafiq/b838e593-9dd8-4deb-829b-9e192e21766e.jpeg"

    # encoded_test = face_recognition.initiate_known_embed(image_path)
    # encode_ref = face_recognition.initiate_known_embed(im2_path)

    # similarity = face_recognition.calc_face_dist(encoded_test, encode_ref)
    # print(f"Cosine distance :", similarity)

    # Perform face detection and recognition
    # boxes, faces = face_recognition.detect_box(img)

    # # Display detected bounding boxes and extracted faces
    # for box, face in zip(boxes, faces):
    #     x1, y1, x2, y2 = [int(a) for a in box]
    #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # cv2.imwrite("output.jpg", img)
    # cv2.imshow("Detected Faces", img)

    # print ("Testing", encoded_test)



