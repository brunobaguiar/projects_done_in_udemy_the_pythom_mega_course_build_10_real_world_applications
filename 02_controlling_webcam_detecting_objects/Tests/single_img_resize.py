import cv2

# Read the image in grey scale with the 0 argument
img = cv2.imread("galaxy.jpg", 0)

# What is the image type?
print(type(img))
# Python interprets the image as a Numpy array:
print(img)
# Image shape
print(img.shape)
# How many dimensions the image have?
print(img.ndim)

# Resize the image by half
resized_image = cv2.resize(img, (int(img.shape[1]/2),int(img.shape[0]/2)))
# Open window with the image
cv2.imshow("Galaxy", resized_image)
# Store resized image
cv2.imwrite("galaxy_resized.jpg", resized_image)
# Time to keep the window open in milliseconds
cv2.waitKey(2000)
# Close all windows
cv2.destroyAllWindows()
