import numpy as np
import cv2
from config import *
from tqdm import tqdm
import os


def sharpen(img):
    """Returns a sharpened version of the image using the sharpening kernel.

    Args:
        img (numpy.ndarray): The image to be sharpened.

    Returns:
        numpy array image: The sharpened image.
    """
    out_img = cv2.filter2D(img, -1, kernel_sharp)

    return out_img


def fusion(img_lap):
    """Returns a list of pyramids of image masks with peak values for fusion

    Args:
        img_lap (list of numpy.ndarray): List of laplacian pyramids.

    Returns:
        list: list of pyramids with peak values for fusion.
    """
    out_pyramid = []

    for n, image in enumerate(img_lap[0]):

        mask = np.uint8(np.zeros((image.shape[0], image.shape[1], image.shape[2])))
        lap_array= np.array([img_lap[i][n] for i in range(len(img_lap))])

        for image in lap_array:
            img_1 = mask.copy()
            mask = np.where(img_1 > image, img_1, image)

        out_pyramid.append(mask)

    return out_pyramid

def gauss_pyr_cv2(img, num_pyramids):
    """Returns a list of gaussian pyramids of image

    Args:
        img (numpy.ndarray): The image to be transformed.
        num_pyramids (integer): The number of pyramids to be created.

    Returns:
        list: list of pyramids.
    """
    img_list = [img]

    for i in range(num_pyramids):
        img = cv2.pyrDown(img)
        img_list.append(img)

    return img_list

def lap_pyr_cv2(img_list, num_pyramids):
    """Returns a list of laplacian pyramids of image

    Args:
        img_list (list of numpy.ndarray): List of gaussian pyramids.
        num_pyramids (integer): The number of pyramids to be created.

    Returns:
        list: list of pyramids.
    """
    laplacian_top = img_list[-1]

    laplacian_pyr = [laplacian_top]
    for i in range(num_pyramids , 0, -1):
        GE = cv2.pyrUp(img_list[i],dstsize=(img_list[i-1].shape[1],img_list[i-1].shape[0]))
        L = cv2.subtract(img_list[i-1],GE)
        laplacian_pyr.append(L)

    return laplacian_pyr

def reconstruct_cv2(out_pyramid, num_pyramids):
    """Returns a reconstructed image from a list of pyramids

    Args:
        out_pyramid (list): List of pyramids.
        num_pyramids (integer): The number of pyramids from which the image is created.

    Returns:
        numpy.ndarray: The reconstructed image.
    """
    out_img = out_pyramid[0]
    for i in range(1, num_pyramids+1):
        out_img = cv2.pyrUp(out_img, dstsize=(out_pyramid[i].shape[1],out_pyramid[i].shape[0]))
        out_img = cv2.add(out_img, out_pyramid[i])
    return out_img

def stacking(image_list):
    """Returns a stacked image from a list of images

    Args:
        image_list (list): List of images.

    Returns:
        numpy.ndarray: The stacked image.
    """
    img_lap = []
    for img in image_list:

        height, width, _ = img.shape

        img_list = gauss_pyr_cv2(img, num_pyramids)
        laplacian_pyr = lap_pyr_cv2(img_list, num_pyramids)

        img_lap.append(laplacian_pyr)

    out_pyramid = fusion(img_lap)
    out_img_reconstruct = reconstruct_cv2(out_pyramid, num_pyramids)

    # out_img = sharpen(out_img_reconstruct)

    return out_img_reconstruct

def orange_hue_change(img):
    """Returns an image with orange hue changed to blue.

    Args:
        img (numpy.ndarray): The image to be transformed.

    Returns:
        numpy.ndarray: The transformed image.
    """    
        
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    previous = 60
    new = 10
    diff_color = new - previous
    hnew = np.mod(h + diff_color, 180).astype(np.uint8)
    hsv_new = cv2.merge([hnew,s,v])
    bgr_new = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

    return bgr_new


def focus_stack_pyramid(images):
    """Returns a stacked image from a list of images

    Args:
        images (list): List of image paths

    Returns:
        numpy.ndarray: The stacked image.
    """
    image_list = []
    for image in tqdm(images, desc="Progress: "):

        img = cv2.imread(image)
        filter_name = os.path.split(image)[-1].split(".")[0].split("_")[-1]

        if filter_name == "O":
            img = orange_hue_change(img)

        image_list.append(img)

    stacked_image = stacking(image_list)

    return stacked_image

#
#   Compute the gradient map of the image
def doLap(image):

    # YOU SHOULD TUNE THESE VALUES TO SUIT YOUR NEEDS
    kernel_size = 5         # Size of the laplacian window
    blur_size = 5           # How big of a kernal to use for the gaussian blur
                            # Generally, keeping these two values the same or very close works well
                            # Also, odd numbers, please...

    blurred = cv2.GaussianBlur(image, (blur_size,blur_size), 0)
    return cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)

#
#   This routine finds the points of best focus in all images and produces a merged result...
#
def focus_stack(image_paths):
    """Returns a stacked image from a list of images"""
    images = []
    for image_path in image_paths:
        images.append(cv2.imread(image_path))
        
    laps = []
    for i in range(len(images)):
        laps.append(doLap(cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY)))

    laps = np.asarray(laps)

    output = np.zeros(shape=images[0].shape, dtype=images[0].dtype)

    abs_laps = np.absolute(laps)
    maxima = abs_laps.max(axis=0)
    bool_mask = abs_laps == maxima
    mask = bool_mask.astype(np.uint8)
    for i in range(0,len(images)):
        output = cv2.bitwise_not(images[i],output, mask=mask[i])
		
    return 255-output