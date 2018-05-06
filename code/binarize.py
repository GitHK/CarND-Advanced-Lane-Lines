from enum import Enum

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def abs_sobel_thresh(channel, is_x_direction=True, sobel_kernel=3, thresh=(0, 255)):
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if is_x_direction:
        sobel = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return sxbinary


def mag_thresh(channel, sobel_kernel=3, mag_thresh=(0, 255)):
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    magnitude_gradient = np.sqrt(sobelx ** 2 + sobely ** 2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * magnitude_gradient / np.max(magnitude_gradient))
    # 5) Create a binary mask where mag thresholds are met
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return sxbinary


def dir_threshold(channel, sobel_kernel=3, dir_thresh=(0.0, np.pi / 2)):
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    dir_binary = np.zeros_like(direction)
    dir_binary[(direction >= dir_thresh[0]) & (direction <= dir_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return dir_binary


class HLS(Enum):
    """ HLS channels for channel extraction"""
    H = 0
    L = 1
    S = 2


class RGB(Enum):
    """ RGB channels for channel extraction"""
    R = 0
    G = 1
    B = 2


class HSV(Enum):
    """ HSV channels for channel extraction"""
    H = 0
    S = 1
    V = 2


def get_hls_image(image):
    """ Converts image to HLS """
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)


def get_hls_channel(image, channel):
    """ Converts an image to HLS and extracts a single channel """
    hls = get_hls_image(image)
    return hls[:, :, channel.value]


def get_hsv_image(image):
    """ Converts image to HSV """
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


def get_hsv_channel(image, channel):
    """ Converts an image to HLS and extracts a single channel """
    hsv = get_hsv_image(image)
    return hsv[:, :, channel.value]


def get_rgb_channel(image, channel):
    """ Converts an image to RGB and extracts a single channel """
    return image[:, :, channel.value]


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def threshold_channel(channel, thresh=(0, 255)):
    """ Apply threshold to a single channel """
    binary = np.zeros_like(channel)
    binary[(channel >= thresh[0]) & (channel <= thresh[1])] = 1
    return binary


def inverse_binarization(channel):
    binary = np.zeros_like(channel)
    binary[~(channel == 1)] = 1
    return binary


def combined_binarization(image, is_debug=True):
    """ Split image into channels and apply a combination of techniques """

    # rgb_r = get_rgb_channel(image, RGB.R)
    # rgb_g = get_rgb_channel(image, RGB.G)
    # rgb_b = get_rgb_channel(image, RGB.B)

    # hls_h = get_hls_channel(image, HLS.H)
    # hls_l = get_hls_channel(image, HLS.L)
    hls_s = get_hls_channel(image, HLS.S)

    # hsv_h = get_hsv_channel(image, HSV.H)
    # hsv_s = get_hsv_channel(image, HSV.S)
    hsv_v = get_hsv_channel(image, HSV.V)

    # TODO ~V from HSV chennel seems to do a good job
    thresh_hsv_v = inverse_binarization(threshold_channel(hsv_v, thresh=(0, 225)))

    # RGB BLOCK

    # abs_sobel_x_rgb_r = inverse_binarization(abs_sobel_thresh(rgb_r, is_x_direction=True, thresh=(0, 20)))      # looks ok but revelas left border of the street
    # abs_sobel_y_rgb_r = inverse_binarization(abs_sobel_thresh(rgb_r, is_x_direction=False, thresh=(0, 100)))    # no, not good at all
    # mag_thresh_rgb_r = mag_thresh(rgb_r, mag_thresh=(50, 100))                                                  # looks bad detecst border of street
    # dir_thresh_rgb_r = dir_threshold(rgb_r, dir_thresh=(0.4, np.pi / 2))                                        # useless
    #
    # abs_sobel_x_rgb_g = inverse_binarization(abs_sobel_thresh(rgb_g, is_x_direction=True, thresh=(0, 20)))      # looks ok but revelas left border of the street
    # abs_sobel_y_rgb_g = inverse_binarization(abs_sobel_thresh(rgb_g, is_x_direction=False, thresh=(0, 100)))    # no, not good at all
    # mag_thresh_rgb_g = mag_thresh(rgb_g, mag_thresh=(50, 100))                                                   # looks bad detecst border of street
    #
    # abs_sobel_x_rgb_b = inverse_binarization(abs_sobel_thresh(rgb_b, is_x_direction=True, thresh=(0, 20)))      # looks ok but revelas left border of the street
    # abs_sobel_y_rgb_b = inverse_binarization(abs_sobel_thresh(rgb_b, is_x_direction=False, thresh=(0, 100)))    # no, not good at all
    # mag_thresh_rgb_b = mag_thresh(rgb_b, mag_thresh=(50, 100))                                                   # looks bad detecst border of street

    # # HLS BLOCK

    # abs_sobel_x_hls_h = inverse_binarization(abs_sobel_thresh(hls_h, is_x_direction=True, thresh=(50, 150)))
    # abs_sobel_y_hls_h = inverse_binarization(abs_sobel_thresh(hls_h, is_x_direction=False, thresh=(0, 100)))
    # mag_thresh_hls_h = mag_thresh(hls_h, mag_thresh=(50, 100))
    # dir_thresh_hls_h = dir_threshold(hls_h, dir_thresh=(-np.pi / 4, np.pi / 4))
    #
    # abs_sobel_x_hls_l = abs_sobel_thresh(hls_l, is_x_direction=True, thresh=(10, 150))
    # abs_sobel_y_hls_l = abs_sobel_thresh(hls_l, is_x_direction=False, thresh=(10, 150))
    # mag_thresh_hls_l = mag_thresh(hls_l, mag_thresh=(50, 100))
    # dir_thresh_hls_l = dir_threshold(hls_l, dir_thresh=(-np.pi / 4, np.pi / 4))

    # TODO: should try and use this one!!!
    abs_sobel_x_hls_s = abs_sobel_thresh(hls_s, is_x_direction=True, thresh=(20, 150))
    # abs_sobel_y_hls_s = abs_sobel_thresh(hls_s, is_x_direction=False, thresh=(10, 150))    # the X one is beter
    # mag_thresh_hls_s = mag_thresh(hls_s, mag_thresh=(50, 100))                              # except test 5 it looks ok
    # dir_thresh_hls_s = dir_threshold(hls_s, dir_thresh=(-np.pi / 4, np.pi / 4))

    # HSV BLOCK
    # abs_sobel_x_hsv_h = abs_sobel_thresh(hsv_h, is_x_direction=True, thresh=(10, 250))                     # no
    # abs_sobel_y_hsv_h = inverse_binarization(abs_sobel_thresh(hsv_h, is_x_direction=False, thresh=(1, 100)))   # bno
    # mag_thresh_hsv_h = mag_thresh(hsv_h, mag_thresh=(50, 100))  # no
    # dir_thresh_hsv_h = dir_threshold(hsv_h, dir_thresh=(-np.pi / 4, np.pi / 4)) #no
    #
    # abs_sobel_x_hsv_s = abs_sobel_thresh(hsv_s, is_x_direction=True, thresh=(20, 200))                  #could work for something
    # abs_sobel_y_hsv_s = abs_sobel_thresh(hsv_s, is_x_direction=False, thresh=(10, 100))         # no
    # mag_thresh_hsv_s = mag_thresh(hsv_s, mag_thresh=(10, 100))  #no
    # dir_thresh_hsv_s = dir_threshold(hsv_s, dir_thresh=(-np.pi / 4, np.pi / 4))
    #
    #
    # TODO: this helps with shadows and yellow lines where there is a lot of nise, some filtering of this channel could do wonders
    abs_sobel_x_hsv_v = abs_sobel_thresh(hsv_v, is_x_direction=True, thresh=(20, 200))  # no
    # abs_sobel_y_hsv_v = abs_sobel_thresh(hsv_v, is_x_direction=False, thresh=(10, 100))     #no
    # mag_thresh_hsv_v = mag_thresh(hsv_v, mag_thresh=(10, 100))  #no
    # dir_thresh_hsv_v = dir_threshold(hsv_v, dir_thresh=(-np.pi / 4, np.pi / 4))


    combined = np.zeros_like(abs_sobel_x_hls_s)
    combined[(abs_sobel_x_hls_s == 1) | (thresh_hsv_v == 1) | (abs_sobel_x_hsv_v == 1)] = 1
    # Other candidates which soldved the problem | (abs_sobel_x_rgb_g == 1) | (abs_sobel_x_rgb_b == 1)

    # combined = abs_sobel_x_hls_s

    if is_debug:
        logger.info("Combined gradients counting %s - %s" % (np.count_nonzero(combined), combined.size))

    return combined
