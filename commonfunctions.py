
import os
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram, rescale_intensity, adjust_gamma
from matplotlib.pyplot import bar
from skimage.color import rgb2gray,rgb2hsv
from sklearn.model_selection import train_test_split
from sklearn import svm
import cv2
from sklearn.neighbors import KNeighborsClassifier
# Convolution:
from scipy.signal import convolve2d
from scipy import fftpack
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math

from skimage.util import random_noise
from skimage.filters import median, gaussian
from skimage.feature import canny, hog

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

# Edges
from skimage.filters import sobel_h, sobel, sobel_v,roberts, prewitt, threshold_otsu,threshold_local
from skimage import transform
from skimage.morphology import skeletonize, thin

from PIL import Image, ImageEnhance

# Show the figures / plots inside the notebook
def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show() 
    

def show_3d_image(img, title):
    fig = plt.figure()
    fig.set_size_inches((12,8))
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(0, img.shape[0], 1)
    Y = np.arange(0, img.shape[1], 1)
    X, Y = np.meshgrid(X, Y)
    Z = img[X,Y]

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0, 8)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title(title)
    plt.show()
    
def show_3d_image_filtering_in_freq(img, f):
    img_in_freq = fftpack.fft2(img)
    filter_in_freq = fftpack.fft2(f, img.shape)
    filtered_img_in_freq = np.multiply(img_in_freq, filter_in_freq)
    
    img_in_freq = fftpack.fftshift(np.log(np.abs(img_in_freq)+1))
    filtered_img_in_freq = fftpack.fftshift(np.log(np.abs(filtered_img_in_freq)+1))
    
    show_3d_image(img_in_freq, 'Original Image')
    show_3d_image(filtered_img_in_freq, 'Filtered Image')


def showHist(img):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imgHist = histogram(img, nbins=256)
    
    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')

def getThreshold(img):
    iut=(img * 255).astype(np.uint8)
    hist = np.zeros(256)
    for i in range(iut.shape[0]):
        for j in range(iut.shape[1]):
            hist[iut[i][j]] += 1
    T_init= round(sum([ i*hist[i] for i in range(256)])/np.cumsum(hist)[-1])
    Tnew = T_init
    while True:
        lower = round(sum([ i*hist[i] for i in range(Tnew)]))/np.cumsum(hist[:Tnew])[-1]
        higher =  round(sum([ i*hist[i] for i in range(Tnew,256)]))/np.cumsum(hist[Tnew:])[-1]
        Tnew = round((lower+higher)/2)
        if Tnew == T_init:
            break
        T_init = Tnew
    return Tnew

def localThreshholding(img: np.ndarray):

    r, c = img.shape
    t_a = getThreshold(img[0:r//2, 0:c//2])
    t_b = getThreshold(img[0:r//2, c//2:c])
    t_c = getThreshold(img[r//2:r, 0:c//2])
    t_d = getThreshold(img[r//2:r, c//2:c])


    image_a = (img[0:r//2, 0:c//2] * 255 < t_a)
    image_b = (img[0:r//2, c//2:c] * 255 < t_b)
    image_c = (img[r//2:r, 0:c//2] * 255 < t_c)
    image_d = (img[r//2:r, c//2:c] * 255 < t_d)

    img[0:r//2, 0:c//2] = image_a
    img[0:r//2, c//2:c] = image_b
    img[r//2:r, 0:c//2] = image_c
    img[r//2:r, c//2:c] = image_d    
    return img
   
