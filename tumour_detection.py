from skimage.io import imread,imsave,imshow
from matplotlib import pyplot as plt
import numpy as np
from skimage.morphology import square, opening , closing , disk ,watershed
from skimage.filters import threshold_otsu , sobel , rank
from skimage.color import rgb2hsv , hsv2rgb,rgb2gray
from skimage.feature import peak_local_max
from skimage import img_as_ubyte
import skimage.filters.rank as skr 

from skimage.segmentation import mark_boundaries , watershed
from skimage.measure import label
from skimage import measure
from skimage.filters.rank import median
from skimage import color

from scipy.ndimage import distance_transform_edt


print(" \n \n\t \t \tWelcome to tumour segmentation tools.\n In this program you can detect tumour within an MRI images.\n" )
print("First you need to enter the name of your MRI image. The images and this program should be in the same repository.\n")




while True :
    try :
        img = input("Enter your image name here, like YourImage.jpg \n")
        img = imread(img)

    except :
            print("Could not read" ,img)
    else : 
        break








img2 = rgb2gray(img)
img2 = (255 * img2).astype(np.uint8)



######## Using watershed transform

#Denoise a bit the image and then compute the Gradient to get detect edges
gradient = skr.gradient(median(img2,disk(4)), disk(1))

# inverse gradient image so that local minima -> local maxima
gradient_i = gradient.max()-gradient 

# Determine automatically markers by computing local
markers_coords = peak_local_max(gradient_i, min_distance=47)

#Labeling the markers >>> region 1, 2, 3.... 
markers = np.zeros_like(img2)
for i, (row,col) in enumerate(markers_coords):
	markers[row,col] = i+1


#From the markers, run Watershed algorithm to grows regions until bassins attributed to other markers
ws = watershed(gradient, markers)

#Compute the size of the tumour by using mask of watershed and region 3 which is the region of tumour,
# by summing the pixel * 0.115 cm/px 
tumour_len = ((ws==3).sum())*0.115

#Save result
imsave('segmented_tumour.jpg', img2*(ws==3))


### Show the result #####

plt.figure(figsize=(15,10))
plt.subplot(1,4,1)
plt.title("Original image")
plt.imshow(img2, cmap=plt.cm.gray)
plt.subplot(1,4,2)
plt.title("Gradient")
plt.imshow(gradient, cmap=plt.cm.gray)
plt.subplot(1,4,3)
plt.title("markers")
plt.imshow(markers, cmap=plt.cm.gray)
plt.subplot(1,4,4)
plt.imshow(img2*(ws==3), cmap=plt.cm.gray)
plt.title("Masked tumour using Watershed")
plt.show()
print("END....")