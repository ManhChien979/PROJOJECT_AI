import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np

face_classifier=cv2.CascadeClassifier('haarcascades_models/haarcascade_frontalface_default.xml')
cam_xuc_model = load_model('cam_xuc.h5')
gioi_tinh_model = load_model('gioi_tinh.h5')
tuoi_model = load_model('tuoi.h5')

#cam_xuc_labels=['angry','disgusted','fearful','happy','neutral','sad','surprised']
cam_xuc_labels=['surprised','angry','disgusted','fearful','happy','neutral','sad']
gioi_tinh_labels = ['female', 'male']
#tuoi_labels = ['20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60']

# load model

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

while True:
    ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.3, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        roi_gray = cv2.resize(roi_gray, (64, 64))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255
        predictions = tuoi_model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        
        tuoi_labels = ['20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60']
        predicted_emotion = tuoi_labels[max_index]
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ############################
      
        roi_color=gray_img[y:y+h,x:x+w]
        roi_color=cv2.resize(roi_color,(48,48),interpolation=cv2.INTER_AREA)
        gender_predict = gioi_tinh_model.predict(np.array(roi_color).reshape(-1,48,48,3))
        gender_predict = (gender_predict>= 0.5).astype(int)[:,0]
        gender_label=gioi_tinh_labels[gender_predict[0]] 
        gender_label_position=(x,y+h+50)
        cv2.putText(test_img,gender_label,gender_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        ##########################
        roi_cx=gray_img[y:y+h,x:x+w]
        roi_cx=cv2.resize(roi_cx,(48,48),interpolation=cv2.INTER_AREA)
        cam_xuc_predict = cam_xuc_model.predict(np.array(roi_cx).reshape(-1,48,48,3))
        cam_xuc_predict = (cam_xuc_predict>= 0.5).astype(int)[:,0]
        cam_xuc_label=cam_xuc_labels[cam_xuc_predict[0]] 
        cam_xuc_label_position=(x,y-30) 
       
        cv2.putText(test_img,cam_xuc_label,cam_xuc_label_position, cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        
        ###########################

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)
    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows