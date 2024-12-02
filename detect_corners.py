import numpy as np
import cv2
import matplotlib.pyplot as plt

def detect_corners(filename):
    image = cv2.imread(filename)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = np.float32(gray_image)

    dst = cv2.cornerHarris(gray_image, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    image[dst > 0.001* dst.max()] = [0, 0, 255]

    return image

if __name__ == '__main__':
    image = detect_corners("dandadan.jpg")
    plt.figure(figsize=(6,8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()