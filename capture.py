# import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
from time import sleep

with PiCamera() as camera:
    camera.resolution = (640, 480)
    camera.start_preview()
    
    sleep(2)
    camera.capture('images/img.jpg')