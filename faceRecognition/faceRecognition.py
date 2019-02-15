from picamera.array import PiRGBArray
from picamera import PiCamera
import os
import cv2
import sys
import pandas


currentDir = "/home/pi/PiProjects/faceRecognition"
facePics = os.path.join(currentDir, "facePics")
modelsDir = os.path.join(currentDir, "models")

# initialize picamera
camera = PiCamera()
camera.resolution = (640,480)
camera.framerate = 16
rawCapture = PiRGBArray(camera, size=camera.resolution)

# 2 seconds warmup for PiCamera
time.sleep(2)

# initialize face detection and recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(modelsDir, sys.args[1], '.yml')
detector = cv2.CascadeClassifier("")

savePath = os.path.join(modelsDir, sys.args[1], '_names.txt')
nameIDs = pandas.read_csv(savePath, delimiter=',',
                            header=None, index_col=0, names=['id','name'])


while True:

    # grab current frame, grayscale image
    frame = camera.capture(rawCapture, format="bgr").array
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # loop over the face detections and draw them on the frame
    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, conf = recognizer.predict(gray[y:y+h,x:x+w])

        if (conf < 100):
            id = nameIDs.iloc[id]['name'].replace('_', ' ').title()
            conf = "  {0}%".format(round(100 - conf))
        else:
            id = "unknown"
            conf = "  {0}%".format(round(100 - conf))

        cv2.putText(frame, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(frame, str(conf), (x+5,y+h-5), font, 1, (255,255,0), 1)


    # show feed
    cv2.imshow("Security Feed", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key is pressed, break from the loop
	if key == ord("q"):
		break

	rawCapture.truncate(0)
