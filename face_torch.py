import os
from os.path import join
from PIL import Image
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.preprocessing import Normalizer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print(model)
print('Loaded Model')

def image_resize(cv_img, k1=640, k2=480):
    x, y = cv_img.shape[0], cv_img.shape[1]
    if x < y:
        x, y = y, x
    return cv2.resize(cv_img, (0, 0), fx=k1/x, fy=k2/y)

def extract_face_from_frame(pixels, required_size=(160, 160)):
    results = mtcnn(pixels, save_path=None, return_prob=False)
    if len(results) == 1:
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        face_array = cv2.resize(face, required_size)
        return face_array
    else:
        return np.array([])  # Return an empty array to match the behavior


def extract_face_from_file(filename, required_size=(160, 160)):
    image = Image.open(filename).convert('RGB')
    face, _ = mtcnn(image, save_path=None, return_prob=False)
    if face is None:
        return None
    face = face.resize(required_size)
    face_array = np.array(face)
    return face_array

def get_embedding(model, face_pixels):
    face_tensor = transforms.ToTensor()(face_pixels).unsqueeze(0).to(device)
    embedding = model(face_tensor).detach().cpu().numpy()[0]
    return embedding

def initiate_known_embed(filename):
    frame = cv2.imread(filename)
    face = extract_face_from_frame(frame)
    if face is None:
        return None
    embed = get_embedding(model, face)
    in_encoder = Normalizer(norm='l2')
    norm_encod = in_encoder.transform([embed])
    return norm_encod

def getFaceEmbeddingsFromImage(face_img):
    embed = get_embedding(model, face_img)
    in_encoder = Normalizer(norm='l2')
    return in_encoder.transform([embed])

def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    probability = np.linalg.norm(face_encodings - face_to_compare, axis=1)
    print(probability)
    return probability

def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.96):
    return (face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)

def get_results_with_detected_face(face, known_embed):
    return compare_faces(known_embed, getFaceEmbeddingsFromImage(face))[0]

def initiate_embeddings_from_file(filename):
    frame = cv2.imread(filename)
    face = extract_face_from_frame(frame)
    if face is None:
        return None
    embed = get_embedding(model, face)
    in_encoder = Normalizer(norm='l2')
    norm_encod = in_encoder.transform([embed])
    return norm_encod

def initiate_embeddings_from_array(frame):
    face = extract_face_from_frame(frame)
    if face is None:
        return None
    embed = get_embedding(model, face)
    in_encoder = Normalizer(norm='l2')
    norm_encod = in_encoder.transform([embed])
    return norm_encod

def setEmbeddingsInFile(personpath, calculate_encodings=[]):
    if not calculate_encodings:
        known_embed = initiate_known_embed(personpath)
    else:
        known_embed = calculate_encodings
    embeddings = known_embed[0]

    first_split = personpath.split('/')

    second_split = first_split[-1].split('.')
    filename = join(os.getcwd(), second_split[0] + '.txt')

    with open(filename, 'w') as filehandle:
        for listitem in embeddings:
            filehandle.write('%s\n' % listitem)

    return second_split[0] + '.txt'

def getEmbeddingsFromFile(path):
    with open(path, 'r') as filehandle:
        embed = [float(line.strip()) for line in filehandle]
    return [embed]

def getResults(frame, embedding_file_path):
    face = extract_face_from_frame(frame)
    if face is None:
        print("Returning false")
        return False
    else:
        file_embedding = getEmbeddingsFromFile(embedding_file_path)
        frame_embedding = getFaceEmbeddingsFromImage(face)
        return compare_faces(file_embedding, frame_embedding)[0]

# Load a sample image and test the functions
sample_image_path = 'path_to_sample_image.jpg'
sample_frame = cv2.imread(sample_image_path)
embedding_file_path = 'path_to_embedding_file.txt'
result = getResults(sample_frame, embedding_file_path)
print("Result:", result)
