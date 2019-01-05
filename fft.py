
import numpy as np
import matplotlib.pyplot as plt
import cv2

def readBgr(fname):
	current_image = cv2.imread(fname)
	current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)
	return current_image

image_stripes = readBgr('images/stripes.jpg')
image_solid = readBgr('images/pink_solid.jpg')

# Display the images
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

# convert to grayscale
gray_stripes = cv2.cvtColor(image_stripes, cv2.COLOR_RGB2GRAY)
gray_solid = cv2.cvtColor(image_solid, cv2.COLOR_RGB2GRAY)

# normalize
normalized_stripes = gray_stripes / 255.0
normalized_solid = gray_solid / 255.0

# perform a fast fourier transform 
# scaled, frequency transform image
def ft_image(norm_image):
    f = np.fft.fft2(norm_image)
    fshift = np.fft.fftshift(f)
    frequency_tx = 20 * np.log(np.abs(fshift))
    return frequency_tx

# Call the function on the normalized images
# and display the transforms
f_stripes = ft_image(norm_stripes)
f_solid = ft_image(norm_solid)

# display the images
# original images to the left of their frequency transform
f, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(20,10))

ax1.set_title('original image')
ax1.imshow(image_stripes)
ax2.set_title('frequency transform image')
ax2.imshow(f_stripes, cmap='gray')

ax3.set_title('original image')
ax3.imshow(image_solid)
ax4.set_title('frequency transform image')
ax4.imshow(f_solid, cmap='gray')

# Read in an image
image = cv2.imread('images/birds.jpg')
# Change color to RGB (from BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# normalize the image
norm_image = gray/255.0

f_image = ft_image(norm_image)

# Display the images
f, (ax1,ax2) = plt.subplots(1, 2, figsize=(20,10))

ax1.imshow(image)
ax2.imshow(f_image, cmap='gray')
