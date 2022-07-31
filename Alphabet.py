import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time
#Imports Done, now reading the csv + accuracy
    #----System Initializing... -> Done----#

x = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)

x_train, x_test, y_train, y_test = train_test_split(x , y, random_state=0, train_size = 0.75, test_size = 0.25)# Try with 10725 and 3575 (0.25 and 0.75 does same thing)
xTrainScaled = x_train/255
xTestScaled = x_test/255

clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(xTrainScaled, y_train)

yPred = clf.predict(xTestScaled)

accuracy = accuracy_score(y_test, yPred)
ay = accuracy_score(yPred, y_test)

print("Accuracy: ", accuracy)

#Accuracy done, now onto the camera part
    #----System Initializing... -> Done----#

cap = cv2.VideoCapture(0)

while (True):
    try:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Changes to greyscale
        
        height, width = gray.shape
        upper_left = (int(width/2 - 56),int(height/2 - 56))
        bottom_right = (int(width/2 + 56),int(height/2 + 56))
        print(upper_left)
        print(bottom_right)
        
        cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 3)
        #Region of Interest

        roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
        
        #print(upper_left[1])
        #print(bottom_right[1])
        #print(upper_left[0])
        #print(bottom_right[0])

        pil = Image.fromarray(roi)

        image = pil.convert('L') #Pixel = single value from 0 to 255
        image_resized = image.resize((28,28))

        real_image = PIL.ImageOps.invert(image_resized) #Flipping image as camera shows mirror image

        minPixel = np.percentile(real_image, 20) #Scalar quantity conversion

        #Scales values between 0 , 255
        real_image_scaled = np.clip(real_image - minPixel, 0, 255) 
        maxPixel = np.max(real_image)

        max_scale = np.asarray(real_image_scaled) / maxPixel #Converts into array

        test = np.array(real_image_scaled).reshape(1,784)
        testPred = clf.predict(test)

        print("Predicted letter is: ", testPred)

        #Shows the image / frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break #Pressing q will end code
    except Exception as e:
        pass

#Destory and release window once done
cap.release()
cv2.destroyAllWindows()