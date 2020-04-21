# Importing the libraries
import numpy as np
import cv2
from PIL import Image

face_classifier = cv2.CascadeClassifier('[PATH TO haarcascade_frontalface_default.xml]'])
eye_classifier = cv2.CascadeClassifier(['[PATH TO haarcascade_eye.xml]')
image = cv2.imread('[PATH TO IMAGE]')
gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray ,1.3 , 9) #location of different faces in file

if faces is ():
    print("no face found")
    
for (x , y , w , h) in faces:
    roi_gray = gray[y:y+h  , x:x+w] #crop face out using indexing
    roi_color = image[y:y+h , x:x+w] #crop face similarily
    cv2.imwrite("Cropped Face!", roi_color)
    

