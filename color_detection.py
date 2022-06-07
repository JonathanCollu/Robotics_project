from PIL import Image
import cv2
#from picamera.array import PiRGBArray
#from picamera import PiCamera
import numpy as np
import time
import sys

np.set_printoptions(threshold=sys.maxsize)

def color_detect(img, color_name):
    # here is the range of H in the HSV color space represented by the color
    # color_dict = {'red':[0,4],'orange':[5,18],'yellow':[22,37],'green':[42,85],'blue':[92,110],'purple':[115,165],'red_2':[165,180]}
    color_dict = {'blue':[0,4],'orange':[5,18],'yellow':[22,37],'green':[42,85],'wrong_blue':[92,110],'purple':[115,165],'red_2':[165,180]}

    # define a 5×5 convolution kernel with element values of all 1.
    kernel_5 = np.ones((5,5), np.uint8) 

    # the blue range will be different under different lighting conditions and can be adjusted flexibly.  
    # H: chroma, S: saturation v: lightness
    # cv2.imshow("video", img)
    
    # in order to reduce the amount of calculation, the size of the picture is reduced to (160,120)
    #resize_img = cv2.resize(img, (160,120), interpolation=cv2.INTER_LINEAR)
    resize_img =  img
    
    #try:
    hsv = cv2.cvtColor(resize_img, cv2.COLOR_BGR2HSV) # convert from BGR to HSV
    #except:
    #    rgb = cv2.cvtColor(resize_img, cv2.COLOR_GRAY2BGR)
    #    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    color_type = color_name
    
    # inRange()：Make the ones between lower/upper white, and the rest black
    mask = cv2.inRange(hsv,np.array([min(color_dict[color_type]), 60, 60]), np.array([max(color_dict[color_type]), 255, 255]))            
    # if color_type == 'red':
    #         mask_2 = cv2.inRange(hsv, (color_dict['red_2'][0],0,0), (color_dict['red_2'][1],255,255))
    #         mask = cv2.bitwise_or(mask, mask_2)

    morphologyEx_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_5,iterations=1) # perform an open operation on the image

    # find the contour in morphologyEx_img, and the contours are arranged according to the area from small to large.
    _tuple = cv2.findContours(morphologyEx_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    # compatible with opencv3.x and openc4.x
    if len(_tuple) == 3:
        _, contours, hierarchy = _tuple
    else:
        contours, hierarchy = _tuple

    color_area_num = len(contours) # Count the number of contours

    if color_area_num > 0:
        for i in contours:    # Traverse all contours
            # Decompose the contour into the coordinates of the upper left corner and the width and height of the recognition object
            x,y,w,h = cv2.boundingRect(i)      

            # Draw a rectangle on the image (picture, upper left corner coordinate, lower right corner coordinate, color, line width)
            if w >= 1 and h >= 1:
            #if w >= 1 and h >= 1: 
                """
                    Because the picture is reduced to a quarter of the original size, 
                    if you want to draw a rectangle on the original picture to circle 
                    the target, you have to multiply x, y, w, h by 4.
                """
                #x = x * 4
                #y = y * 4
                #w = w * 4
                #h = h * 4
                try:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)  # Draw a rectangular frame
                except: pass
                # cv2.putText(img,color_type,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)  # Add character description

    return img, mask, morphologyEx_img

def interpret_image(ball_color, border_color, image):
    _, ball_mask, _ = color_detect(image, ball_color)
    _, border_mask, _ = color_detect(image, border_color)
    ball_mask = np.round_(ball_mask / 255)
    border_mask = np.round_(border_mask / 255)
    mask = np.stack([ball_mask, border_mask])
    # composite_image = ball_mask + 2 * border_mask
    return mask

if __name__ == "__main__":
    src_img = cv2.imread('img-1.png')
    """
    img,img_2,img_3 =  color_detect(src_img,'red')  # Color detection function
    cv2.imshow("video", img)    # OpenCV image show
    cv2.imshow("mask", img_2)    # OpenCV image show
    cv2.imshow("morphologyEx_img", img_3)    # OpenCV image show
    """
    
    mask = interpret_image("green", "red", src_img)
    print(mask.shape)
    # print(list(set(i for j in ball_mask for i in j)))
    
    ball_mask = np.array(mask[0] * 255, dtype=np.uint8)
    cv2.imwrite("ball_mask.png", ball_mask)
    border_mask = np.array(mask[1] * 255, dtype=np.uint8)
    cv2.imwrite("border_mask.png", border_mask)
