import os
import logging
import cv2
import numpy as np

from code.constants import OUTPUT_IMAGES_DIR
from code.utils import files_in_directory, get_module_directory, load_pickle_from_file, dump_pickle_to_file, \
    create_directory_if_missing, file_exist_in_path, bgr_to_rgb, undistort_image, save_images_with_title

logger = logging.getLogger(__name__)


def calibrate_camera(calibration_dir):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d points in real world space
    img_points = []  # 2d points in image plane.

    # Make a list of calibration images
    calibration_image_names = files_in_directory(calibration_dir, supported_extensions=["jpg"])

    # Search all chessboard corners and append them if found
    for calibration_image_name in calibration_image_names:
        calibration_image = cv2.imread(calibration_image_name)
        gray = cv2.cvtColor(calibration_image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret == True:
            obj_points.append(objp)
            img_points.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    # return calibration parameters
    return ret, mtx, dist, rvecs, tvecs


def save_calibration_chessboard_images(calibration_dir_path, calibration_images_dir, mtx, dist,
                                       out_image_suffix='calibrated'):
    """
    Calibrates the chessboard images and saves them to the output folder inside a new subfolder with
    the same name as the provided input folder.

    If an image was already created it will not be replaced. To start a new generation delete old crated images.

    :param calibration_dir_path: path to where the source images are stored
    :param calibration_images_dir: name of the directory in which the source images are stored
    """
    out_files_dir = os.path.join(get_module_directory(), '..', OUTPUT_IMAGES_DIR)

    create_directory_if_missing(out_files_dir)

    out_path = os.path.join(out_files_dir, calibration_images_dir)
    create_directory_if_missing(out_path)

    calibration_image_names = files_in_directory(calibration_dir_path, supported_extensions=["jpg"], full_path=False)

    for calibration_image_name in calibration_image_names:
        # create image if it does not exist
        out_file_path = os.path.join(out_path, '%s_%s' % (out_image_suffix, calibration_image_name))
        in_file_path = os.path.join(calibration_dir_path, calibration_image_name)
        if not file_exist_in_path(out_file_path):
            # make and save the output in this path
            logger.info('Reading %s' % in_file_path)
            img = bgr_to_rgb(cv2.imread(in_file_path))
            undistorted = undistort_image(img, mtx, dist)
            save_images_with_title(img, undistorted, 'Original', 'Undistorted', out_file_path)


def get_calibration(calibration_images_dir):
    """
    Returns a previously create calibration. If missing, creates and saves a new one together with the new
    calibration images in the output directory
    """

    calibration_dir_path = os.path.join(get_module_directory(), '..', calibration_images_dir)
    calibration_params_file = os.path.join(calibration_dir_path, 'calibrated_params.pickle')

    pickled_content = load_pickle_from_file(calibration_params_file)

    if pickled_content is None:
        logger.info('No calibration parameters found, GENERATING...')
        calibration_params = calibrate_camera(calibration_dir_path)
        dump_pickle_to_file(calibration_params, calibration_params_file)

        # extract parameters and save images in output directory
        _, mtx, dist, _, _ = calibration_params
        save_calibration_chessboard_images(calibration_dir_path, calibration_images_dir, mtx, dist)

        pickled_content = calibration_params

    logger.info('Calibration parameters loaded')
    return pickled_content
