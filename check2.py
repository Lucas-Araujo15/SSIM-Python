import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('images/3.jpg')  # queryImage
img2 = cv2.imread('images/3.jpg')  # trainImage

# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params, search_params)

des1 = np.float32(des1)
des2 = np.float32(des2)

matches = flann.knnMatch(des1, des2, k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0, 0] for i in range(len(matches))]

good_points = []

threshold = 0.7

# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < threshold*n.distance:
        matchesMask[i] = [1, 0]
        good_points.append(m)

percentage = 100 / 500
print(str(len(good_points) * percentage) + '% de semelhança')

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=0)

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

plt.imshow(img3,), plt.show()
