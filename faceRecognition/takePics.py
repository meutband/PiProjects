from picamera.array import PiRGBArray
from picamera import PiCamera
import os
import cv2
import sys


currentDir = "/home/pi/PiProjects/faceRecognition"
facePics = os.path.join(currentDir, "facePics")

# initialize picamera
camera = PiCamera()
camera.resolution = (640,480)
camera.framerate = 16
rawCapture = PiRGBArray(camera, size=camera.resolution)

# 2 seconds warmup for PiCamera, initialize face detection
time.sleep(2)

detector = cv2.CascadeClassifier('')

print("Please enter the name(s) of the people to take their pictures")
print("First and Last Name or First and Last Initial).")
print("Separate the names by a comma")
names = input("Names: ")

for name in names:

total = len(os.listdir(os.path.))

while True:

    # grab current frame
    frame = camera.capture(rawCapture, format="bgr").array
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

	# loop over the face detections and draw them on the frame
	for (x, y, w, h) in rects:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `k` key was pressed, write the *original* frame to disk
	# so we can later process it and use it for face recognition
	if key == ord("k"):
        p = os.path.join(facePics, "{}.png".format(str(total).zfill(4))])
		cv2.imwrite(p, frame)
		total += 1

	# if the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break
