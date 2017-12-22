import cv2  
import numpy as np
import matplotlib.pyplot as plot
cap = cv2.VideoCapture(0)

def chanageColor():
    print 123

cv2.namedWindow('image')
cv2.createTrackbar('R','image',0,255,chanageColor())
cv2.createTrackbar('G','image',0,255,chanageColor())
cv2.createTrackbar('B','image',0,255,chanageColor())
while(1):
    # get a frame
    ret, frame = cap.read()
    # show a frame
    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
    lower_red = np.array([0,0,0])
    upper_red = np.array([255,255,120])
    mask = cv2.inRange(hsv,lower_red,upper_red)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    opened = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    closed = cv2.morphologyEx(opened,cv2.MORPH_OPEN,kernel) 
    #ret, binary = cv2.threshold(blurred,90,255,cv2.THRESH_BINARY)  
  
    # gradX = cv2.Sobel(binary, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    # gradY = cv2.Sobel(binary, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # # subtract the y-gradient from the x-gradient
    # gradient = cv2.subtract(gradX, gradY)
    # gradient = cv2.convertScaleAbs(gradient)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
    # closed = cv2.morphologyEx(gradient,cv2.MORPH_CLOSE,kernel)
    # closed = cv2.dilate(closed, None, iterations=3)
    # closed = cv2.erode(closed, None, iterations=2)
    #opened = cv2.morphologyEx(opened,cv2.MORPH_OPEN,kernel)
    im2,contours, hierarchy = cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
    cv2.drawContours(frame,contours,-1,(0,0,255),3)  
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    # opened = cv2.morphologyEx(gradient,cv2.MORPH_CLOSE,kernel)
    # closed = cv2.morphologyEx(opened,cv2.MORPH_OPEN,kernel)
    # blur and threshold the image
    #blurred = cv2.blur(gradient, (9, 9))
    #(_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
    #im2,contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
    #cv2.drawContours(frame,contours,-1,(0,0,255),3)  
    cv2.imshow("capture", frame)
    cv2.imshow("test", closed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break