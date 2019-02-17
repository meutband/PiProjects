import os
import cv2
import sys
import numpy


currentDir = "/home/pi/PiProjects/faceRecognition"
facePics = os.path.join(currentDir, "facePics")
modelsDir = os.path.join(currentDir, "models")

# initialize face detection and recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("/home/pi/opencv/data/haarcascades/haarcascade_frontalface_default.xml")

# initialize list of name folders to grab their images
names = os.listdir(facePics)

# list each image path and each id for recognizer
imagePaths = []
ids = []
faceSamples = []

# create list name with id for identification in face recognition
nameIDs = []

# loop through name folders
for i, name in enumerate(names):

	# append id and name
	nameIDs.append((i, name))
	# get list of images in name folder
	namePics = os.listdir(os.path.join(facePics, name))

	# loop through list of pics
	for pic in namePics:
		# add imagePath and id for faceRecognition
		imagePaths.append(os.path.join(facePics, name, pic))
		ids.append(i)

for image in imagePaths:

	# read image and convert to gray scale
	frame = cv2.imread(image)
	orig = frame.copy()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# find face in images, cut image to box of fce
	faces = detector.detectMultiScale(gray)

	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		cv2.imshow("Press 's' to save face; Press 'd' to delete face", frame)
		key = cv2.waitKey() & 0xFF

		if key == ord("s"):
			faceSamples.append(gray[y:y+h,x:x+w])
			frame = orig

		if key == ord("d"):
			frame = orig
			continue


recognizer.train(faceSamples, numpy.array(ids))

saveModel = os.path.join(modelsDir, sys.argv[1]+'.yml')
recognizer.write(saveModel)

savePath = os.path.join(modelsDir, sys.argv[1]+'_names.txt')
numpy.savetxt(savePath, numpy.array(nameIDs), delimiter=',', fmt='%s')
