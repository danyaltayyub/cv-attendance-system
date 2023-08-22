import cv2
import torch
import pandas as pd
from facenet_pytorch import InceptionResnetV1, MTCNN
import numpy as np
from tqdm import tqdm
from types import MethodType
# from get_encodings import get_encoding, encode, mtcnn, resnet
from training import train_model, encode_from_cropped, get_processed_img
from record import record, initiate, search_entry, close_file
from sklearn.metrics.pairwise import cosine_similarity
# from pdb import set_trace as debug


VIDEO_PATH = "rtsp://admin:JGPHAN@192.168.0.118:554"
file = initiate()

# def encode_from_cropped(cropim):
#     cropim = cropim.cuda()
#     embed = resnet (cropim.unsqueeze(0))
#     return embed.detach()

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

### load model

tdevice = torch.device('cuda' if torch.cuda.is_available else 'cpu')
mtcnn = MTCNN(
  image_size=160, margin=0, keep_all=True, thresholds=[0.6, 0.7, 0.7], min_face_size=40, 
  factor=0.709, post_process=True , device= tdevice
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(tdevice)
mtcnn.detect_box = MethodType(detect_box, mtcnn)
BATCH_SIZE = 8
FRAME_SKIP = 5

def frame_processing(image, saved_data, thres = 0.7):
    min_dist = ""
     # cv2.imshow("test", img0)
    
    embedding_list = saved_data[0] # getting embedding data
    name_list = saved_data[1] # getting list of names
    batch_boxes, cropped_images = mtcnn.detect_box(image)
    # batch_boxes, cropped_images = batch_boxes.cuda(), cropped_images.cuda()
    if cropped_images is not None:
        for box, cropped in zip(batch_boxes, cropped_images):
            x, y, x2, y2 = [int(x) for x in box]
            img_embedding = encode_from_cropped(cropped)
            # batch_images.append(resnet(cropped.unsqueeze(0)).detach())
            # batch_images = torch.cat(batch_images, dim=0)
            # batch_embeddings = batch_images
            
            
            dist_list = [] # list of matched distances, minimum distance is used to identify the person

            for idx, emb_db in enumerate(embedding_list):
                # ar1 = np.linalg.norm(img_embedding)
                # ar2 = np.linalg.norm(emb_db)
                # dota = np.dot(ar1 , ar2)
                dist = cosine_similarity(img_embedding.reshape(1,-1), emb_db.reshape(1,-1))
                print(name_list[idx], dist[0][0])
                dist_list.append(dist[0][0])
                name_list.append(name_list[idx])
            print("-------------")


    
            idx_min = dist_list.index(max(dist_list))
            print (idx_min)            # idx_min = dist_list[0]


            # if dist_list[idx_min] < thres:
            if dist_list[idx_min] >= thres:
                min_key = name_list[idx_min]
                # min_key = name_list[0]

                min_dist = str(round(dist_list[idx_min],2))
                # min_dist = dist_list[0] 
                
            elif dist_list[idx_min] < thres:
                # min_key = name_list[idx_min]
                # min_key = name_list[0]
                min_key = 'Undetected'

            # min_dist = str(round(dist_list[idx_min],2))


            # dists = [[(e1 - e2).norm().item() for e2 in img_embedding] for e1 in all_people_faces]
            # print (dists)
            # print(pd.DataFrame(dists, index=names))

            cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(image, (x, y2 - 70), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(image, min_key, (x + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(image, min_dist, (x + 30, y2 - 30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

            if search_entry(min_key, file):
                continue
            else:
                record(min_key, file)

        return image



def detect(cam, thres=1):
    vdo = cv2.VideoCapture(cam)
    data = torch.load('data.pt') # loading data.pt file
    # frames = []
    # i=1
    while vdo.grab():
        try:
            _, img0 = vdo.read()
            # frames.append(Image.fromarray(img0))
            # if i % FRAME_SKIP == 0:
            img0 = frame_processing(img0 , data)
            # i = 1
            # i +=1
            if (img0 is not None):
                cv2.imshow("output", img0)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break
        except Exception as e:
            print("image is", img0)
            print(str(e))
            print("\n\n\n\n Reinitializing...")
            vdo = cv2.VideoCapture(cam)

if __name__ == "__main__":
    
    faces, names = train_model('saved')
    detect(VIDEO_PATH)
    # img = cv2.imread("oa.jpg")
    # img2 = frame_processing(img)
    # cv2.imwrite("ff.jpg", img2)
    # close_file(file)