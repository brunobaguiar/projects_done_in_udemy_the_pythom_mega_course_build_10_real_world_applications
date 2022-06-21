import cv2
import glob

# glob return a list of all file paths that match a specific pattern
images = glob.glob("*.jpg")
print(images)

for image in images:
    img = cv2.imread(image, 0)
    re = cv2.resize(img, (100, 100))
    cv2.imshow("Hey", re)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    cv2.imwrite("resized_"+image, re)
