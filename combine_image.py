from PIL import Image
import numpy as np

cuboid_mask = Image.open('./images/img_cuboid1_16.png')
border_mask = Image.open('./images/img_border1_16.png')
cuboid_array = np.asarray(cuboid_mask)
border_array = np.asarray(border_mask)

new_image = cuboid_array + border_array
image = Image.fromarray(new_image)
image.save('combine_masks1_16.png')
