from picamera.array import PiRGBArray
from picamera import PiCamera
import os
import cv2
import datetime
import time

# initialize picamera
camera = PiCamera()
camera.resolution = (640,480)
camera.framerate = 16
rawCapture = PiRGBArray(camera, size=camera.resolution)

# 2 seconds warmup for PiCamera
time.sleep(2)

# start average for motion change
avg = None

while True:

    # grab current frame
    frame = camera.capture(rawCapture, format="bgr").array

    # grayscale and blur image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # set first frame as average
    if avg is None:
		avg = gray.copy().astype("float")
		rawCapture.truncate(0)
		continue

    # calculate average between previos average,
    # calculate differece between average and currentFrame
    cv2.accumulateWeighted(gray, avg, 0.5)
	frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    # calculate threshold of difference, dilate, and find contours of threshold
    thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255,
		cv2.THRESH_BINARY)[1]
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)

    # loop through contours
    for c in cnts:

		# if the contour is too small, ignore it
		if cv2.contourArea(c) < 5000:
			continue

		# compute the bounding box , draw it on the frame
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# draw the timestamp on the frame
    timestamp = datetime.datetime.now()
	ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
	cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
		0.35, (0, 0, 255), 1)

    # show feed
    cv2.imshow("Security Feed", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key is pressed, break from the loop
	if key == ord("q"):
		break

	rawCapture.truncate(0)
