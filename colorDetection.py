# import the necessary packages
import numpy as np
import argparse
import cv2


######## TO LOAD THE SCRIPT RUN THE FILE WITH THE '--image' ARGUMENT FOLLOWED BY THE IMAGE PATH 
# python colorDetection --image images\test_robot1.jpg

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")   
args = vars(ap.parse_args())
image = cv2.imread(args["image"])


'''color scheme : [B, G, R]'''

boundaries = [
    # ([17, 15, 100], [50, 56, 200]),
	# ([0, 0, 100], [50, 50, 255]), # red
    # ([0, 0, 100], [60, 60, 255]), # redish
    # ([120, 0, 200], [170, 50, 255]),  # orange
    # ([80, 0, 150], [200, 100, 255]), # wide orange
    ([0, 180, 190], [115, 255, 255]) # yellow
]


for (lower, upper) in boundaries:
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask = mask)
    cv2.imshow("images", np.hstack([image, output]))
    cv2.imwrite('images/test_colors.jpg', output)
    cv2.waitKey(0)
