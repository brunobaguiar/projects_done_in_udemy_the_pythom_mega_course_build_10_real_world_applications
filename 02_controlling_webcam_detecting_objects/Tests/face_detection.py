import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread("news.jpg")

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# scaleFactor is the accuracy of the search in %, as lesser more accuracy
# 1.05 is more precise than 1.5
faces = face_cascade.detectMultiScale(gray_img,
                                      scaleFactor=1.05,
                                      minNeighbors=5)

for x, y, w, h in faces:
    # draw a rectangle with face coordinates, color as tuple, and int for width of rectangle line
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

print(type(faces))
# faces will return [[upper left  coord, upper left y coord, width, height]]
print(faces)

resized = cv2.resize(img,(int(img.shape[1]/3),int(img.shape[0]/3)))

cv2.imshow('Gray', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
