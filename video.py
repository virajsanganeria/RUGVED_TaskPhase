import cv2 as cv
import numpy as np

vid = cv.VideoCapture("/home/viraj/Documents/opencv1/pothole detection/pothole.webm")
while True:
    vi, frame = vid.read()
    f1 = frame[25:570,50:850]
    cv.imshow("Pothole2", f1)
    gray = cv.cvtColor(f1, cv.COLOR_BGR2GRAY)
    blur = cv.blur(gray, (25,25))
    blur1 = cv.GaussianBlur(blur, (25,25),11)
    blur2 = cv.medianBlur(blur1,21)
    blur3 = cv.bilateralFilter(blur2,11, 25,25)
    cv.imshow(".", blur3)
    threshold, thresh = cv.threshold(blur3,140,155,cv.THRESH_BINARY)
    cv.imshow("Threshold", thresh)
    contours,hier = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    blank = np.zeros(f1.shape, dtype='uint8')
    canny = cv.Canny(thresh,125,175)
    cv.drawContours(f1, contours, -1, (0,0,255),3)
    cv.imshow("Final image", f1)



    if(cv.waitKey(1)& 0xFF==ord('q')):
        break
vid.release()

cv.waitKey(0)