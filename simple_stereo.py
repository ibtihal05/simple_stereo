import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math

img1=cv.imread('img1.jpg',0)
img2=cv.imread('img2.jpg',0)
b=10



sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

good = []
for m,n in matches:
    if m.distance < 0.5*n.distance:
        good.append([m])
        
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good, None, flags=2)
plt.imshow(img3)
plt.show()

#the calibration matrix of the camera which took the given images
f=[[2.81890930e+03, 0.00000000e+00, 5.04377414e+02],
 [0.00000000e+00, 1.86070797e+04, 4.02523951e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]


list_x=[]
list_y=[]
list_z=[]

for i in range(len(good)):
    # Get the match num i
    match = good[i][0]
    # Get the keypoints corresponding to the match
    kp1_idx = match.queryIdx
    kp2_idx = match.trainIdx
    # Get the coordinates of the keypoints
    u1,v1 = kp1[kp1_idx].pt
    u2,v2 = kp2[kp2_idx].pt
    z=(b*f[0][0])/abs((u1-u2))
    x=b*(u1-f[0][2])/abs(u1-u2)
    y=b*f[0][0]*(u1-f[0][2])/(f[1][1]*abs(u1-u2))
    #y=b*f[0][0]*(v1-f[0][2])/(f[1][1]*abs(u1-u2))
    list_x.append(x)
    list_y.append(y)
    list_z.append(z)
    
print(list_z)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(list_x, list_y, list_z, c='b', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend(loc='upper left')
plt.show()

