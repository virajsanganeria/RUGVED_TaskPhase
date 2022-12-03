import cv2 as cv
import numpy as np

img = cv.imread("/home/viraj/Documents/opencv1/pothole detection/pothole.jpg")
cv.imshow("Pothole", img)
resize =img[25:576,50:850]
gray = cv.cvtColor(resize,cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)
blur = cv.blur(gray, (21,21))
blur1 = cv.GaussianBlur(blur, (21,21),0)
blur2 = cv.medianBlur(blur1,21)
blur3 = cv.bilateralFilter(blur2,15,30,30)
cv.imshow("djfv", blur3)
Threshold, thresh = cv.threshold(blur3, 100, 250, cv.THRESH_BINARY)
cv.imshow("Simple thresholding", thresh)
blank = np.zeros(blur3.shape, dtype= 'uint8')
contours, hier = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
blank = np.zeros(resize.shape, dtype="uint8")
canny = cv.Canny(thresh,125,175)
cv.drawContours(blank,contours,-1, (0,0,255),1)
cv.imshow("Pothole final", blank)






cv.waitKey(0)