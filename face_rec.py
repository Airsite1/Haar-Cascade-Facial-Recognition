# Importing the libraries
import numpy as np
import cv2
from PIL import Image

face_classifier = cv2.CascadeClassifier('/Users/sanjay/Desktop/CODE/Python/Covid19-Face-Mask-Classifier-with-webscraped-images/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('/Users/sanjay/Desktop/CODE/Python/Covid19-Face-Mask-Classifier-with-webscraped-images/haarcascade_eye.xml')
image = cv2.imread('/Users/sanjay/Desktop/IMG_3415.JPG')
gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray ,1.3 , 9) #location of different faces in file

if faces is ():
    print("no face found")
    
for (x , y , w , h) in faces:
    roi_gray = gray[y:y+h  , x:x+w] #crop face out using indexing
    roi_color = image[y:y+h , x:x+w] #crop face similarily
    cv2.imwrite("OH", roi_color)
    print('ran')

