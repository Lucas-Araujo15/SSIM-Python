from skimage.transform import resize
from skimage.metrics import structural_similarity
import cv2


def orb_sim(img1, img2):

    orb = cv2.ORB_create()

    kp_a, desc_a = orb.detectAndCompute(img1, None)
    kp_b, desc_b = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(desc_a, desc_b)

    similar_regions = [i for i in matches if i.distance < 50]
    if len(matches) == 0:
        return 0
    return len(similar_regions) / len(matches)


def structural_sim(img1, img2):

    sim, diff = structural_similarity(img1, img2, full=True)
    return sim


img00 = cv2.imread('images/monkey_distorted.jpg', 0)
img01 = cv2.imread('images/monkey_rotated.jpg', 0)

img1 = cv2.imread('images/3.jpg', 0)
img2 = cv2.imread('images/BSE_noisy.jpg', 0)
img3 = cv2.imread('images/2.jpg', 0)
img4 = cv2.imread('images/different_img.jpg', 0)

orb_similarity = orb_sim(img1, img3)

print("A similaridade usando ORB é: ", orb_similarity)

img5 = resize(img3, (img1.shape[0], img1.shape[1]),
              anti_aliasing=True, preserve_range=True)

ssim = structural_sim(img1, img5)
print("A similaridade usanso SSIM é: ", ssim)
