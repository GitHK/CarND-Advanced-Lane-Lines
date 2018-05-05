"""
Transforms the camera image with a perspective transformation, because the camera image is always the same
"""
import cv2
import os
import numpy as np
import matplotlib.pylab as plt

from code.calibration import get_calibration
from code.constants import TEST_IMAGES_DIR, CALIBRATION_IMAGES_DIR
from code.utils import get_module_directory, bgr_to_rgb, undistort_image


def warp_image(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, Minv


def unwarp_image(img, Minv):
    img_size = (img.shape[1], img.shape[0])
    unwraped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)
    return unwraped


# defined here for speed boost, after they are computed
udacity_cam_src = np.float32([
    [585, 460],
    [203, 720],
    [1127, 720],
    [695, 460]
])

requested_dst = np.float32([
    [320, 0],
    [320, 720],
    [960, 720],
    [960, 0],
])


def wrap_udacity_camera(img):
    """
    Points are mapped on a quadrilateral of starting form top left angle in clockwise direction.
    Points are named: A, B, C, D
    """
    warped, Minv = warp_image(img, udacity_cam_src, requested_dst)
    return warped, Minv


def show_image_with_pints(img, points):
    """ Plots the image with the corresponding points on top of it """
    plt.plot()
    plt.imshow(bgr_to_rgb(img))
    for point in points:
        plt.plot(*point, '.')
    plt.show()


def perspective_tuning(image_name):
    image_path = os.path.join(get_module_directory(), '..', TEST_IMAGES_DIR, image_name)
    img = cv2.imread(image_path)

    img_width = img.shape[1]
    img_height = img.shape[0]
    print('original', img_width, img_height)

    udacity_cam_src = np.float32([
        [604, 443],
        [673, 443],
        [1054, 690],
        [248, 690]
    ])

    udacity_A = udacity_cam_src[0]
    udacity_B = udacity_cam_src[1]
    udacity_C = udacity_cam_src[2]
    udacity_D = udacity_cam_src[3]

    max_x = max(udacity_B[0], udacity_C[0])
    min_x = min(udacity_A[0], udacity_D[0])

    width = max_x - min_x
    offset_x = (img_width - width) / 2

    print("Width offset", width, offset_x)

    requested_dst = np.float32([
        [offset_x, 0],
        [offset_x + width, 0],
        [offset_x + width, img_height],
        [offset_x, img_height],
    ])


    udacity_cam_src = np.float32([
        [585, 460],
        [203, 720],
        [1127, 720],
        [695, 460]
    ])

    requested_dst = np.float32([
        [320, 0],
        [320, 720],
        [960, 720],
        [960, 0],
    ])


    calibration_params = get_calibration(CALIBRATION_IMAGES_DIR)
    # later usage
    ret, mtx, dist, rvecs, tvecs = calibration_params
    undistorted_image = undistort_image(img, mtx, dist)
    show_image_with_pints(undistorted_image, udacity_cam_src)
    # cv2.imwrite('undistorted.jpg', undistorted_image)

    print(udacity_cam_src)
    print(requested_dst)

    warped, Minv = warp_image(undistorted_image, udacity_cam_src, requested_dst)

    show_image_with_pints(warped, requested_dst)
    # cv2.imwrite('wrapped.jpg', warped)

    unwraped = unwarp_image(undistorted_image, Minv)

    show_image_with_pints(unwraped, udacity_cam_src)
    # cv2.imwrite('unwrapped.jpg', unwraped)


if __name__ == '__main__':
    perspective_tuning('straight_lines1.jpg')
