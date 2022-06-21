import cv2
from datetime import datetime

import pandas
import pandas as pd

# variable to store first frame as static background
first_frame = None
status_list = [None, None]
times = []
df = pandas.DataFrame(columns=["Start", "End"])

# to access camera, first camera available will be 0, then 1, and so on.
# To access video files, just change the number for the path string
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    # starts reading the camera frame
    check, frame = video.read()
    status = 0

    # grayscale frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blur the image on gray
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Capture the first frame as the background
    if first_frame is None:
        first_frame = gray
        # stop this iteration of the loop and restart
        continue

    # Compare the initial frame with current frame
    delta_frame = cv2.absdiff(first_frame, gray)
    # transform all moving objects in white, considering moving objects color 30 or less on delta_frame
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    # Dilate areas, to remove shadows interference on white
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # Find contours of the image
    (cnts, _) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over objects with area > 10000
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        # if something enter the view, change the status to 1
        status = 1
        # Draw a rectangle on found moving objects with area > 1000
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # record the status per frame
    status_list.append(status)

    # If the status chance, record the timestamp
    if status_list[-1] == 1 and status_list[-2] == 0 or \
            status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    # Show frame on window
    cv2.imshow("Gray Frame", gray)
    cv2.imshow("Delta Frame", delta_frame)
    cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Color Frame", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        if status == 1:
            times.append(datetime.now())
        break
print(status_list)
print(times)

for i in range(0, len(times), 2):
    df = df.append({"Start": times[i], "End": times[i + 1]}, ignore_index=True)

df.to_csv("times.csv")

video.release()
cv2.destroyAllWindows()
