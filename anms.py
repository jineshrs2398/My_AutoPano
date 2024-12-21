#from detect_corners import detect_corners
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import peak_local_max
def detect_corners(filename):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    gray_image = np.float32(image)

    corners = cv2.cornerHarris(gray_image, 2, 3, 0.04)
    stddev = np.std(corners)
    corners = (corners - corners.mean())/stddev

    return corners

def corners_local_maxima(corners,factor):
    std_dev = np.std(corners)
    local_max_corners = peak_local_max(corners, min_distance=3, threshold_abs=std_dev*factor)
    return local_max_corners

# def detect_corners(image):
#     image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
#     harris_response = cv2.cornerHarris(image, 2, 3, 0.04)
#     corners = np.argwhere(harris_response > 0.01* harris_response)  # argswhere - you are getting the indices
#     corner_strength = harris_response[corners[:, 0], corners[:, 1]] # this is the value aka strength using the indices
#     return corners, corner_strength

def apply_anms(corner_strength, corners, max_corners):
    anms_radii = []
    for i, (x, y) in enumerate(corners):
        min_radius = float('inf')
        for j, (x2, y2) in enumerate(corners):
            if corner_strength[j] > corner_strength[i]:
                distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                min_radius = min(min_radius, distance)
        anms_radii.append((min_radius, x, y))

    anms_radii.sort(reverse=True, key=lambda x:x[0])
    selected_corners = np.array([(x,y) for _, x, y in anms_radii[:max_corners]])
    return selected_corners



corners = detect_corners("dandadan.jpg")
max_corners = 300
factor = 1
local_max_corners = corners_local_maxima(corners , factor)
corner_strength = corners[local_max_corners[:,0], local_max_corners[:,1]]
anms_corners = apply_anms(corner_strength, local_max_corners, max_corners)

image = cv2.imread("dandadan.jpg")
plt.figure(figsize=(6, 8))
for x, y in anms_corners:
    cv2.circle(image, (int(y), int(x)), radius=3, color=(0, 255, 0), thickness=-1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
# if __name__ == '__main__':
#     image = 'dandadan.jpg'
#     corners, corner_strength = detect_corners(image)
#     plt.figure(figsize=(6,8))
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     plt.axis("off")
#     plt.show()