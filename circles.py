import cv2
import numpy as np


def circleDetection(imagePathRed, imagePathAll):
    
    # ONLY RED CIRCLES
    img = cv2.imread(imagePathRed, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray, (3, 3))
    
    # Apply Hough transform on the blurred image.
    minDist = 20
    param1 = 500 #500
    param2 = 200 #200 #smaller value-> more false circles
    minRadius = 1
    maxRadius = 50 #10

    # red_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, param2 = 30, minRadius = 1, maxRadius = 40)    #default params.
    red_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1.2, 75, param1 = 50, param2 = 50, minRadius = 1, maxRadius = 50)  #the images X1 params.
    # red_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1.5, 75, param1 = 50, param2 = 50, minRadius = 5, maxRadius = 0)    #the images X2 params.

    # red_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    
    # Draw circles that are detected.
    if red_circles is not None:
        
        red_circles = np.uint16(np.around(red_circles))
    
        for pt in red_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
    
            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r, (0, 255, 0), 2)
    
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
            cv2.imshow("Detected Circle", img)
            cv2.waitKey(0)

        cv2.imwrite('images/img_colors_redBalls.jpg', img)

    else:
        print('No red detected circles...')


    # THE REST OF THE CIRCLES (EXCLUDING RED)
    img = cv2.imread(imagePathAll, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray, (3, 3))
    
    rest_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1.2, 75, param1 = 50, param2 = 50, minRadius = 1, maxRadius = 50)  
    
    if rest_circles is not None:
        rest_circles = np.uint16(np.around(rest_circles))
        for pt in rest_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
    
            cv2.circle(img, (a, b), r, (0, 255, 0), 2)
    
            cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
            cv2.imshow("Detected Circle", img)
        
        cv2.waitKey(0)
        cv2.imwrite('images/img_colors_otherBalls.jpg', img)

    else:
        print('No other detected circles...')


    return red_circles, rest_circles


red_circles, rest_circles = circleDetection('images/test_colors.jpg', 'images/colorfulBalls.png')   # pathing with the mask image and the orignal image
print(red_circles)  # positon within the image of the red circles (or target color)
print(rest_circles) # rest colored circles