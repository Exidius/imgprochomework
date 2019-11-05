import cv2
import numpy as np
import os
import math
from objloader_simple import *

MIN_MATCHES = 15
model = cv2.imread('model.jpg', 0)

cam = cv2.VideoCapture(0)
ret, cap = cam.read()



# ORB keypoint detector
orb = cv2.ORB_create()              
# create brute force  matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  
# Compute model keypoints and its descriptors
kp_model, des_model = orb.detectAndCompute(model, None)  
# Compute scene keypoints and its descriptors
kp_frame, des_frame = orb.detectAndCompute(cap, None)
# Match frame descriptors with model descriptors
matches = bf.match(des_model, des_frame)
# Sort them in the order of their distance
matches = sorted(matches, key=lambda x: x.distance)

if len(matches) > MIN_MATCHES:
    # draw first 15 matches.
    cap = cv2.drawMatches(model, kp_model, cap, kp_frame,
                          matches[:MIN_MATCHES], 0, flags=2)
    # show result
    cv2.imshow('frame', cap)
    cv2.waitKey(0)
else:
    print ("Not enough matches have been found - %d/%d")
