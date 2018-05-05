"""
Defines all elements required for the pipeline to work properly.
"""
import logging
from collections import deque

from code.binarize import combined_binarization
from code.calibration import get_calibration
from code.constants import CALIBRATION_IMAGES_DIR
from code.line_detector import combined_line_detector
from code.overaly import overlay_detected_lane, print_curvature_and_distance
from code.perspective import wrap_udacity_camera
from code.utils import undistort_image, binary_image_to_gray

logger = logging.getLogger(__name__)

LINE_DETECTION_HISTORY_MAX_ELEMENTS = 10

calibration_params = None
debug_mode = True
debug_output = False


def set_debug_mode_options(is_debug, use_debug_output):
    """
    Sets some debug options.
    :param is_debug: enables logging of all messages if True else only error messages and above are shown
    :param use_debug_output: if True all the intermediate steps will be returned in an array else only the
                            final processed image will be provided
    :return: array of intermediate outputs with labels `[(raw_data, name), ...]` or a single image `raw_data`
    """
    global debug_mode
    global debug_output

    debug_mode = is_debug
    debug_output = use_debug_output
    log_level = logging.INFO if is_debug else logging.ERROR
    logger.setLevel(log_level)


labeled_results = deque()  # stores output if debug_output is True

detection_history = deque(maxlen=LINE_DETECTION_HISTORY_MAX_ELEMENTS)


def reset_pipeline():
    """ Initializes global variables used to keep track of the detections made"""
    global detection_history
    detection_history = deque(maxlen=LINE_DETECTION_HISTORY_MAX_ELEMENTS)


def initialize_labeled_results():
    global labeled_results
    labeled_results = deque()


def append_result_with_label(result, label):
    """
    Appends a result with a label to the global labeled_results
    :param result: output image
    :param label: image label
    """
    global labeled_results
    global debug_output
    if debug_output:
        labeled_results.append((result, label))


def get_labeled_results():
    global labeled_results
    return labeled_results


def process_raw(raw_image_data, is_video_frame=True):
    initialize_labeled_results()

    logger.info('STARTED')

    logger.info('1. camera calibration')
    global calibration_params
    global debug_mode

    # If not already calibrated, calibrate the first time this pipeline is started.
    # Store calibration in memory for later usages, this boosts pipeline speed significantly
    if calibration_params is None:
        calibration_params = get_calibration(CALIBRATION_IMAGES_DIR)
    # later usage
    ret, mtx, dist, rvecs, tvecs = calibration_params

    logger.info('2. distortion correction')
    undistorted_image = undistort_image(raw_image_data, mtx, dist)
    append_result_with_label(undistorted_image, '1_undistorted')

    logger.info('3. create threshold binary image')
    binarized_image = combined_binarization(undistorted_image, is_debug=debug_mode)
    append_result_with_label(binary_image_to_gray(binarized_image), '2_binarized')

    logger.info('4. perspective transformation')
    wraped_binarized, Minv = wrap_udacity_camera(binarized_image)
    append_result_with_label(binary_image_to_gray(wraped_binarized), '3_warped_binarized')

    logger.info('5. lane line search')
    detected_debug, lane_info = combined_line_detector(wraped_binarized, debug_mode, is_video_frame, detection_history)
    append_result_with_label(detected_debug, '4_detected_lines_debug')

    logger.info('6. output detected line')
    output_image = overlay_detected_lane(wraped_binarized, undistorted_image, Minv, lane_info.left_fitx,
                                         lane_info.right_fitx, lane_info.ploty)

    logger.info('7. Overlaying text information')
    output_image = print_curvature_and_distance(output_image, lane_info.left_curverad_m, lane_info.right_curverad_m,
                                                lane_info.lane_position, lane_info.lane_width)

    append_result_with_label(output_image, 'final')
    logger.info('FINISHED')

    global debug_output
    return get_labeled_results() if debug_output else output_image
