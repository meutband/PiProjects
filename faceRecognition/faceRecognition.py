from picamera.array import PiRGBArray
from picamera import PiCamera
import os
import cv2
import sys


# initialize picamera
camera = PiCamera()
camera.resolution = (640,480)
camera.framerate = 16
rawCapture = PiRGBArray(camera, size=camera.resolution)

# 2 seconds warmup for PiCamera
time.sleep(2)
