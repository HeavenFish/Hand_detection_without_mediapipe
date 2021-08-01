import cv2 as cv
import numpy as np

capture = cv.VideoCapture(0)
cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
while(capture.isOpened()):
    check, img = capture.read()

    #cv.imshow('frame', img)
    kernel = np.ones((2, 2), np.uint8)
    #HSv
    hsvim = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([0, 15, 0], dtype="uint8")
    upper = np.array([17, 170, 255], dtype="uint8")
    HSVmask = cv.inRange(hsvim, lower, upper)

    HSVmask = cv.dilate(HSVmask, kernel, iterations=1)
    cv.imshow('hsv', HSVmask)
    #HSVmask = cv.morphologyEx(HSVmask, cv.MORPH_OPEN, np.ones((3,3), np.uint8))

    #YCrCb
    img_YCrCb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    YCrCb_mask = cv.inRange(img_YCrCb, (0, 135, 85), (255, 180, 135))
    #YCrCb_mask = cv.erode(YCrCb_mask, kernel, iterations=2)
    #YCrCb_mask = cv.morphologyEx(YCrCb_mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))

    global_mask = cv.bitwise_and(YCrCb_mask, HSVmask)
    #global_mask = cv.GaussianBlur(global_mask, (3,3), 0)
    #global_mask = cv.morphologyEx(global_mask, cv.MORPH_OPEN, np.ones((4,4), np.uint8))
    cv.imshow('global mask', global_mask)

    HSV_result = cv.bitwise_not(HSVmask)
    YCrCb_result = cv.bitwise_not(YCrCb_mask)
    result = global_result=cv.bitwise_not(global_mask)
    cv.imshow('result1', result)
    result = cv.dilate(result, kernel, iterations=5)
    result = cv.erode(result, kernel, iterations=14)

    #cv.imshow("1_HSV.jpg", HSV_result)
    #cv.imshow("2_YCbCr.jpg", YCrCb_result)
    #cv.imshow('result', result)
    #cv.imshow('img', img)

    thresh = cv.bitwise_not(result)
    #thresh = result

    #remove face
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_detect = cascade.detectMultiScale(gray, 1.2, 5)
    for (x, y, w, h) in face_detect:
        # cv.rectangle(img, (x,y), (x+w+10,y+h+10), (255,255,0), thickness=5)
        # print(x,y,w,h)
        thresh[y:y+w+w, x:h+x] = 20
        __, thresh = cv.threshold(thresh, 50, 255, cv.THRESH_BINARY)
    cv.imshow('thresh', thresh)

    #drawContour
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #cv.imshow('thresh2', thresh)
    print('len =', len(contours))
    if(len(contours) != 0):
        contours = max(contours, key=cv.contourArea)
        #print(contours)
        if(cv.contourArea(contours) >= 10000):
            cv.drawContours(img, [contours], -1, (0,255,0), 2)
            x,y,w,h = cv.boundingRect(contours)
            cv.rectangle(img, (x,y), (x+w,y+h), (255, 255, 0), thickness=5)
        print(cv.contourArea(contours))
    #hull = cv.convexHull(contours)
    #cv.drawContours(img, [hull], -1, (0, 255, 255), 2)
    #hull = cv.convexHull(contours, returnPoints=False)
    #defects = cv.convexityDefects(contours, hull)
    cv.imshow('contours', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        cv.destroyAllWindow()
        break