from PIL import Image
import sys
import cv2
import numpy as np
from PIL import Image
from copy import deepcopy
from picamera import PiCamera

np.set_printoptions(threshold=sys.maxsize)


def color_detect(img, color_name):
    # here is the range of H in the HSV color space represented by the color
    color_dict = {'red_1': [0, 10], 'orange': [11, 28], 'yellow': [29, 34], 'green': [
        25, 90], 'blue': [86, 130], 'purple': [131, 155], 'red_2': [156, 180]}

    # define a 5×5 convolution kernel with element values of all 1.
    kernel_5 = np.ones((5, 5), np.uint8)

    # the blue range will be different under different lighting conditions and can be adjusted flexibly.
    # H: chroma, S: saturation v: lightness

    # in order to reduce the amount of calculation, the size of the picture is reduced to (160,120)
    resize_img = img

    hsv = cv2.cvtColor(resize_img, cv2.COLOR_BGR2HSV)  # convert from BGR to HS
    color_type = color_name

    # inRange()：Make the ones between lower/upper white, and the rest black
    mask = cv2.inRange(hsv, np.array([min(color_dict[color_type]), 60, 60]), np.array(
        [max(color_dict[color_type]), 255, 255]))

    # perform an open operation on the image
    morphologyEx_img = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel_5, iterations=1)

    # find the contour in morphologyEx_img, and the contours are arranged according to the area from small to large.
    _tuple = cv2.findContours(
        morphologyEx_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # compatible with opencv3.x and openc4.x
    if len(_tuple) == 3:
        _, contours, hierarchy = _tuple
    else:
        contours, hierarchy = _tuple

    color_area_num = len(contours)  # Count the number of contours

    if color_area_num > 0:
        for i in contours:    # Traverse all contours
            # Decompose the contour into the coordinates of the upper left corner and the width and height of the recognition object
            x, y, w, h = cv2.boundingRect(i)

            # Draw a rectangle on the image (picture, upper left corner coordinate, lower right corner coordinate, color, line width)
            if w >= 1 and h >= 1:
                try:
                    # Draw a rectangular frame
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                except:
                    pass

    return img, mask, morphologyEx_img


def interpret_image(ball_color, border_color, image):
    _, ball_mask, _ = color_detect(image, ball_color)
    _, border_mask, _ = color_detect(image, border_color)
    ball_mask = np.round_(ball_mask / 255)
    Image.fromarray(np.array(ball_mask * 255, dtype=np.uint8)
                    ).save('test_ball.png')
    border_mask = np.round_(border_mask / 255)
    Image.fromarray(np.array(border_mask * 255, dtype=np.uint8)
                    ).save('test_border.png')
    mask = np.stack([ball_mask, border_mask])
    return mask


def border_is_too_close(border_mask):
    # find the lowest point of the border
    noiseless_img = deepcopy(border_mask)
    noiseless_img = np.array(noiseless_img * 255)
    noiseless_img = np.uint8(noiseless_img)
    noiseless_img = cv2.Canny(noiseless_img, 1, 100)
    noiseless_img = cv2.fastNlMeansDenoising(noiseless_img, h=20)
    cv2.imwrite("noiseless_border_mask.png", noiseless_img)
    border_bottom = -1
    i = 0

    for row in noiseless_img:
        if 255 in row:
            border_bottom = i
            i += 1

    # Considered image76.png (bottom_border=71) to be outside of the allowed
    # area. When bottom_border <= 71, the border is too close
    if border_bottom <= 35:
        print(border_bottom)
        return True
    return False


if __name__ == "__main__":
    camera = PiCamera()
    camera.resolution = (480, 368)
    camera.framerate = 24
    camera.start_preview()
    img = np.empty((480 * 368 * 3,), dtype=np.uint8)
    camera.capture(img, 'rgb')
    src_img = img.reshape((368, 480, 3))

    mask = interpret_image("green", "purple", src_img)
    print(mask.shape)

    ball_mask = np.array(mask[0] * 255, dtype=np.uint8)
    cv2.imwrite("ball_mask.png", ball_mask)
    border_mask = np.array(mask[1] * 255, dtype=np.uint8)
    cv2.imwrite("border_mask.png", border_mask)
