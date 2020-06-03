#Python: 3.5.4 
#IPython: 7.9.0
#numpy: 1.18.1
#matplotlib: 3.0.3
#openCV: 4.2.0
#scikit-image: 0.15.0

# import the necessary packages
from skimage import measure
import matplotlib.pyplot as plt
import numpy as np
import cv2


def mse(imageA, imageB):
# the 'Mean Squared Error' between the two images is the sum of the squared difference between the two images;
# NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

# return the MSE, the lower the error, the more "similar" the two images are
    return err
def compare_images(imageA, imageB, title):
# compute the mean squared error and structural similarity index for the images
    m = mse(imageA, imageB)
    s = measure.compare_ssim(imageA, imageB)
# setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
# show first image
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title(title)
    plt.imshow(imageA, cmap = plt.cm.gray)
    plt.axis("on")
# show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap = plt.cm.gray)
    plt.axis("on")
# show the images
#plt.show()

# load the images 
original = cv2.imread("images/original.jpg")
predicted = cv2.imread("images/predicted.jpg")
# convert the images to grayscale
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
predicted = cv2.cvtColor(predicted, cv2.COLOR_BGR2GRAY)

# initialize the figure
fig = plt.figure()
images = ("Original", original), ("Predicted", predicted)
# loop over the images
for (i, (name, image)) in enumerate(images):
# show the image
    ax = fig.add_subplot(1, 3, i + 1)
    ax.set_title(name)
    plt.imshow(image, cmap = plt.cm.gray)
    plt.axis("off")
# show the figure
plt.show()
# compare the images
compare_images(original, original, "Original vs. Original")
compare_images(original, predicted, "Original vs. Predicted")
plt.show()