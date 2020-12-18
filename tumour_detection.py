from skimage.io import imread,imsave,imshow
from matplotlib import pyplot as plt
import numpy as np
from skimage.morphology import square, opening , closing , disk ,watershed
from skimage.filters import threshold_otsu , sobel , rank
from skimage.color import rgb2hsv , hsv2rgb,rgb2gray
from skimage.feature import greycomatrix, greycoprops , peak_local_max , canny
from skimage import img_as_ubyte
import skimage.filters.rank as skr 

from skimage.segmentation import mark_boundaries , watershed
from skimage.measure import label
from skimage import measure
from skimage.filters.rank import median
from skimage    import color
from scipy import ndimage as ndi 

print(" \n \n\t \t \tWelcome to tumour segmentation tools.\n In this program you can detect tumour within an MRI images.\n" )
print("First you need to enter the name of your MRI image. The images and this program should be in the same repository.\n")




while True :
    try :
        image = input("Enter your image name here, like YourImage.jpg \n")
        img = imread(image)

    except :
            print("Could not read" ,image)
    else : 
        break
 



rgb_img = rgb2gray(img)

### Using canny detection to get the edges of the contours
edge = canny(rgb_img , sigma = 4)

#### calculate the distance 
dt = ndi.distance_transform_edt(~edge)


#### find automatic position's markers using the local maxima
local_max = peak_local_max(dt, indices=False , min_distance=5)



#### Labeled the marked by enumerate them
markers = label(local_max)

#### Use watershed transfom on an inverting distance 
labels = watershed(-dt , markers) 



### Show the result #####
plt.figure()
plt.subplot(1,3,1)
plt.title("Original image")
plt.imshow(img  , cmap = plt.cm.gray )
plt.subplot(1,3,2)
plt.title("Regions image")
plt.imshow(mark_boundaries(img,labels) , cmap = plt.cm.gray)
plt.subplot(1,3,3)
plt.title("Labeled image")
plt.imshow(color.label2rgb(labels , image=img  )   )

plt.show()

print("END....")