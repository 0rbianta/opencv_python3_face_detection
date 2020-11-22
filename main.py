import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier("face.xml") #Load cascade model. This includes informations about positive and negative images to detect image

human_image = cv2.imread('human.webp') #load image as matrix(mat) data

human_image_gray_painted = cv2.cvtColor(human_image, cv2.COLOR_BGR2GRAY) #paint to gray to detect objects with CascadeClassifier


face = face_classifier.detectMultiScale(human_image_gray_painted,1.3,5) # detect objects over the gray painted image

print(face) # this will show us where is the face location (x,y,z,h)

for (x,y,w,h) in face: # so we have x,y and weight(w), height(h). x,y will be a point to show where will rectangle draw start and weight(w), height(h) will define the rectangles size.
    human_image = cv2.rectangle(human_image,(x,y),(w+x,h+y),(47,11,48),3) #RGB = Red,Green,Blue but opencv uses "BGB" so the new combination is BLUE GREEN RED! 47 blue, 11 green, 48 red
    #note 47,11,48 on BGR means purple color. Last arguvment(3) means how our rectangle will be bold.

    #                                 This is our rectangle bold value ¬
    #                This is rectangles BGR(BLUE,GREEN,RED) value  ¬    |
    #                                                              V    V
    #human_image = cv2.rectangle(human_image,(x,y),(w+x,h+y),(47,11,48),3)
    #                                  ^      ^       ^
    #                                         |        ∟-------------------¬
    #           data to overwrite rectangle   |                            |            
    # point that drawing will start(rectangles left top point)             |
    # x will be our start point to start drawin rectangles weight and y will be out start point to start drawing height

    
    
cv2.imshow('OUTPUT DATA WINDOW',human_image) # Output drawed image with a window
cv2.waitKey(0) #Wait for keypress to continue
cv2.destroyAllWindows() # close windows
