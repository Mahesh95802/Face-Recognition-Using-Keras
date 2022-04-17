import cv2
import os

def registerNewFace(db):
    print("Starting the Registration Process...")
    name = input("Name of the new Face: ")
    face_cascade = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")
    captureDevice = cv2.VideoCapture(0) 
    pathtrain = db+"\\Train\\"+name
    pathtest = db+"\\Test\\"+name
    os.mkdir(pathtrain)
    os.mkdir(pathtest)
    path = pathtrain
    img_id = 0
    flag = 0
    while(True):
        ret, frame = captureDevice.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)
        if(img_id >= 160):
            flag += 1
            path = pathtest
        for (x, y, w, h) in faces:
            print(x, y, w, h)
            color = (255, 0, 0)
            stroke = 2
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, stroke)
            cropped_face = frame[y:y+h,x:x+w]
            if cropped_face is not None:
                img_id+=1
                face = cv2.resize(cropped_face, (200,200))
                file_name_path = path+"\\"+str(img_id)+'.jpg'
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(img_id-flag), (200,200), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2 )
                cv2.imshow("Cropped_Face", face)
        cv2.imshow('frame', frame)
        if (cv2.waitKey(20) & 0xFF == ord('q')) or int(img_id)==200:
            break
    if int(img_id)==200:
        print("Images have been Captured.")
    captureDevice.release()
    cv2.destroyAllWindows() 
    print("Images have been collected.")