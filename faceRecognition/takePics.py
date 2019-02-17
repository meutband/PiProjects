from picamera.array import PiRGBArray
from picamera import PiCamera
import os
import cv2
import sys
import time


currentDir = "/home/pi/PiProjects/faceRecognition"
facePics = os.path.join(currentDir, "facePics")

# initialize picamera
camera = PiCamera()
camera.resolution = (640,480)
camera.framerate = 16
rawCapture = PiRGBArray(camera, size=camera.resolution)

# 2 seconds warmup for PiCamera, initialize face detection
time.sleep(2)
detector = cv2.CascadeClassifier('/home/pi/opencv/data/haarcascades/haarcascade_frontalface_default.xml')

# get user input of name(s) to take pictures of
print("Please enter the name(s) of the people to take their pictures")
print("First and Last Name or First and Last Initial).")
print("Separate the names by a comma")
names = input("Names: ").split(',')
names = [name.strip().replace(" ", "_").lower() for name in names]

for name in names:

	# initialize path to name folder
	namePath = os.path.join(facePics, name)

	# if name folder doesnt exists, makedir
	if not os.path.exists(namePath):
		os.makedirs(namePath)

	# get total number of pics in namedir, (incase adding to)
	total = len(os.listdir(namePath))

	while True:

		# grab current frame
		camera.capture(rawCapture, format="bgr")
		frame = rawCapture.array.copy()
		orig = frame.copy()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

		# loop over the face detections and draw them on the frame
		for (x, y, w, h) in rects:
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# show frame, wait for key
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# `k` pressed, save pic for facedetection
		if key == ord("k"):
			p = os.path.join(namePath, "{}.png".format(str(total).zfill(3)))
			cv2.imwrite(p, orig)
			total += 1

		# if the `q` key was pressed, break from the loop
		elif key == ord("q"):
			break

		rawCapture.truncate(0)
