import cv2
import glob
import imutils

def stitch_images(image_paths):
    images = [cv2.imread(img) for img in image_paths]
    images = [imutils.resize(img, width=800) for img in images]

    stitcher = cv2.Stitcher_create()
    status, stitched = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        return stitched
    else:
        print("Error during stitching:", status)
        return None
    
image_paths = glob.glob("*.jpg")
result = stitch_images(image_paths)

if result is not None:
    cv2.imshow("Panaroma", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("stiching failed")