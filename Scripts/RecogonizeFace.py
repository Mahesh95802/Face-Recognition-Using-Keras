import numpy as np
import cv2
import os
import pickle
from PIL import Image
from keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db = os.path.join(BASE_DIR, "DataBase")

def recogonizeFace():
    model = load_model('FaceRecognitionModel.h5')
    print("Model Successfully Loaded")
    print("Starting to Load Labels....")
    labels = {}
    with open("labels.pickle","rb") as f:
        labels = pickle.load(f)
        labels = {v:k for k,v in labels.items()}
    print(labels)
    face_cascade = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")
    captureDevice = cv2.VideoCapture(0)
    while(True):
        ret, frame = captureDevice.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)
        for (x, y, w, h) in faces:
            print(x, y, w, h)
            color = (255, 0, 0)
            stroke = 2
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, stroke)
            cropped_face = frame[y:y+h,x:x+w]
            if cropped_face is not None:
                face = cv2.resize(cropped_face, (200,200))
                img_data = Image.fromarray(face, 'RGB')
                img_array = np.array(img_data)
                img_array = np.expand_dims(img_array, axis=0)
                model_out = model.predict(img_array)
                print(model_out)
                print("----------------------------------------------------------------")
                #print(model_out[0])
                font = cv2.FONT_HERSHEY_SIMPLEX
                color = (255, 0, 0)
                stroke = 2
                print(labels[np.argmax(model_out)])
                cv2.putText(frame,labels[np.argmax(model_out)],(x,y),font,1,color,stroke,cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if (cv2.waitKey(20) & 0xFF == ord('q')):
            break