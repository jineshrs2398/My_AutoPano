import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import peak_local_max
import imutils
from pathlib import Path

def stitchImagePairs(img0, img1, H):
    image0 = img0.copy()
    image1 = img1.copy()

    h0,w0 = img0.shape[:2]
    h1,w1 = img1.shape[:2]

    points_on_image0 = np.float32([[0,0],[0,h0],[w0,h0],[w0,0]]).reshape(-1,1,2)
    points_on_image0_transformed = cv2.perspectiveTransform(points_on_image0,H)

    points_on_image1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    points_on_merged_images = np.concatenate((points_on_image0_transformed, points_on_image1), axis=0)
    points_on_merged_images_ = []

    for p in range(len(points_on_merged_images)):
        points_on_merged_images_.append(points_on_merged_images[p].ravel())

    points_on_merged_images_ = np.array(points_on_merged_images_)

    x_min, y_min = np.int0(np.min(points_on_merged_images_,axis=0))
    x_max, y_max = np.int0(np.max(points_on_merged_images_,axis=0))

    canvas_width = x_max - x_min
    canvas_height = y_max - y_min

    H_translate = np.array([[1,0,-x_min],[0,1,-y_min],[0,0,1]]) #-ve translate to bring transformed image0 to image 1
    images0_transformed_and_stitched = cv2.warpPerspective(image0,np.dot(H_translate,H),(
        canvas_width, canvas_height))

    images_stitched = images0_transformed_and_stitched.copy()
    images_stitched[-y_min:-y_min+h1,-x_min:-x_min+w1] = image1

    indices = np.where(image1==[0,0,0])
    y = indices[0] + -y_min
    x = indices[1] + -x_min

    images_stitched[y,x] = images0_transformed_and_stitched[y,x]

    return images_stitched
def estimateHomography(image1, image2):

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, tree =5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return M

def Warping(Img, Homography, NextShape):
    nH, nW, _ = Img.shape
    Borders = np.array([[0, nW, nW, 0], [0, 0, nH, nH], [1, 1, 1, 1]])
    BordersNew = np.dot(Homography, Borders)
    Ymin = min(BordersNew[1]/BordersNew[2])
    Xmin = min(BordersNew[0]/BordersNew[2])
    Ymax = max(BordersNew[1]/BordersNew[2])
    Xmax = max(BordersNew[0]/BordersNew[2])
    if Ymin < 0:
        MatChange = np.array(
            [[1, 0, -1 * Xmin], [0, 1, -1 * Ymin], [0, 0, 1]])
        Hnew = np.dot(MatChange, Homography)
        h = int(round(Ymax - Ymin)) + NextShape[0]
    else:
        MatChange = np.array(
            [[1, 0, -1 * Xmin], [0, 1, Ymin], [0, 0, 1]])
        Hnew = np.dot(MatChange, Homography)
        h = int(round(Ymax + Ymin)) + NextShape[0]
    w = int(round(Xmax - Xmin)) + NextShape[1]
    sz = (w, h)
    PanoHolder = cv2.warpPerspective(Img, Hnew, dsize=sz)
    return PanoHolder, int(Xmin), int(Ymin)


def create_panorama(img_array):
    image_array = img_array.copy()
    N = len(image_array)
    image0 = image_array[0]
    j = 0
    for i in range (1, N):
        j = j + 1
        image1 = image_array[i]
        image_pair = [image0, image1]
        H = estimateHomography(image0, image1)
        stitched_image = stitchImagePairs(image0, image1, H)
        image0 = stitched_image
    return image0

image1 = imutils.resize(cv2.imread("IMG_2877.jpg"),width=400)
image2 = imutils.resize(cv2.imread("IMG_2878.jpg"),width=400)
image3 = imutils.resize(cv2.imread("IMG_2879.jpg"),width=400)
image4 = imutils.resize(cv2.imread("IMG_2880.jpg"),width=400)
image5 = imutils.resize(cv2.imread("IMG_2881.jpg"),width=400)
image6 = imutils.resize(cv2.imread("IMG_2882.jpg"),width=400)
image7 = imutils.resize(cv2.imread("IMG_2883.jpg"),width=400)


# image1 = cv2.imread("image/1.jpg")
# image2 = cv2.imread("image/2.jpg")
# image3 = cv2.imread("image/3.jpg")
# List to store the images
# images = [image1, image2, image3, image4]
# images = [imutils.resize(image,width=400) for image in images]

# images = []
# folder_path = Path("image")
# for img_path in folder_path.glob("*"):
#     img = cv2.imread(str(img_path))
#     resized_img = imutils.resize(img, width=250)
#     images.append(resized_img)

# N = len(images)
# N_images = len(images)
# N_first_half = round(N_images/2)
# N_second_half = N_images - N_first_half

# while N_images is not 2:
#     merged_images = []
#     for n in range(0, N_first_half,2):
images51 = [image5, image1]
panorama51 = create_panorama(images51)


# image56_1 = [panorama51, panorama61]
# panorama56_1 = create_panorama(image56_1)

image2002 = [image6, panorama51]
panorama1 = create_panorama(image2002)

image52 = [image5, image2]
panorama52 = create_panorama(image52)
images51 = [image1, panorama52]
panorama51 = create_panorama(images51)
images61 = [image6, panorama51]
panorama61 = create_panorama(images61)

images42 = [image4, panorama61]
panorama42 = create_panorama(images42)
images342 = [image3, panorama42]
panorama342 = create_panorama(images342)
images7342 = [image7, panorama342]
panorama7342 = create_panorama(images7342)







cv2.imshow('Panorama56_1', panorama7342)
cv2.imwrite('My PanoStitching.png',panorama7342 )
#cv2.imshow('Panoramay', panorama52)

#cv2.imshow('Panoramax', panoramax)


cv2.waitKey(0)
cv2.destroyAllWindows()

