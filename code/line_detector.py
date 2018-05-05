import logging
from collections import deque

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Y_ARRAY_INDEX_OF_BOTTOM_ELEMENT = 0


class LaneInfo:
    def __init__(self):
        self.left_fit = None
        self.right_fit = None

        self.left_fitx = None
        self.right_fitx = None
        self.ploty = None

        # curve radius of the left and right curves
        self.left_curverad_m = None
        self.right_curverad_m = None

        # position from the center of the lane in meters  < 0 left >0 right
        self.lane_position = None

        # width of the lane in meters
        self.lane_width = None

        # keep track of minimum and maximum coordinate of y axis
        self.min_left_y = None
        self.max_left_y = None
        self.min_right_y = None
        self.max_right_y = None


def full_line_search(wraped_binarized, ym_per_pix=30 / 720, xm_per_pix=3.7 / 700, with_debug_image=True):
    TOTAL_VERTICAL_STRIDES = 10
    # Set minimum number of pixels found to recenter window
    MIN_PIXELS_TO_RECENTER = 10
    DRAWN_CURVE_LINE_WIDTH = 4  # width of final curve in pixels

    scan_window_width = int(wraped_binarized.shape[1] * 0.10)
    half_scam_window_width = int(scan_window_width / 2)
    scan_window_height = int(wraped_binarized.shape[0] / TOTAL_VERTICAL_STRIDES)

    debug_output = np.dstack((wraped_binarized, wraped_binarized, wraped_binarized)) * 255

    if with_debug_image:
        logger.info("Search params (strides, window width, window height): (%s, %s, %s)" % (
            TOTAL_VERTICAL_STRIDES, scan_window_width, scan_window_height))

    histogram = np.sum(wraped_binarized[wraped_binarized.shape[0] // 2:, :], axis=0)

    midpoint = np.int(histogram.shape[0] // 2)
    left_search_base = np.argmax(histogram[:midpoint])
    right_search_base = np.argmax(histogram[midpoint:]) + midpoint

    nonzero = wraped_binarized.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    total_height = wraped_binarized.shape[0]

    r_bottom_center = [right_search_base, total_height]
    l_bottom_center = [left_search_base, total_height]

    left_lane_indexes = deque()
    right_lane_indexes = deque()

    for stride in range(TOTAL_VERTICAL_STRIDES):
        # upper and lower margin of the scanning window
        y_bottom = total_height - (stride + 1) * scan_window_height
        y_top = total_height - stride * scan_window_height

        # left search window right & left boundaries
        l_search_window_left_x = l_bottom_center[0] - half_scam_window_width
        l_search_window_right_x = l_bottom_center[0] + half_scam_window_width

        # right search window right & left boundaries
        r_search_window_left_x = r_bottom_center[0] - half_scam_window_width
        r_search_window_right_x = r_bottom_center[0] + half_scam_window_width

        # Draw the windows on the visualization image
        if with_debug_image:
            cv2.rectangle(debug_output, (l_search_window_left_x, y_bottom), (l_search_window_right_x, y_top),
                          (0, 255, 0), 2)
            cv2.rectangle(debug_output, (r_search_window_left_x, y_bottom), (r_search_window_right_x, y_top),
                          (0, 255, 0), 2)

        left_indexes = ((nonzero_y >= y_bottom) & (nonzero_y < y_top) &
                        (nonzero_x >= l_search_window_left_x) & (nonzero_x < l_search_window_right_x)).nonzero()[0]
        right_indexes = ((nonzero_y >= y_bottom) & (nonzero_y < y_top) &
                         (nonzero_x >= r_search_window_left_x) & (nonzero_x < r_search_window_right_x)).nonzero()[0]

        # Append these indices to the lists
        left_lane_indexes.append(left_indexes)
        right_lane_indexes.append(right_indexes)

        # If you found > MIN_PIXELS_TO_RECENTER, recenter next window on their mean position
        if len(left_indexes) > MIN_PIXELS_TO_RECENTER:
            l_bottom_center[0] = np.int(np.mean(nonzero_x[left_indexes]))
        if len(right_indexes) > MIN_PIXELS_TO_RECENTER:
            r_bottom_center[0] = np.int(np.mean(nonzero_x[right_indexes]))

    left_lane_indexes = np.concatenate(left_lane_indexes)
    right_lane_indexes = np.concatenate(right_lane_indexes)

    # Extract left and right line pixel positions
    left_x = nonzero_x[left_lane_indexes]
    left_y = nonzero_y[left_lane_indexes]
    right_x = nonzero_x[right_lane_indexes]
    right_y = nonzero_y[right_lane_indexes]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    ploty = np.linspace(0, wraped_binarized.shape[0] - 1, wraped_binarized.shape[0]).astype(np.int)

    # Generate x and y values for plotting
    left_fitx = (left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]).astype(np.int)
    right_fitx = (right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]).astype(np.int)

    lane_position = get_lane_position(wraped_binarized, left_fitx, right_fitx, xm_per_pix)
    lane_width = get_lane_width(left_fitx, right_fitx, xm_per_pix)

    if with_debug_image:
        logger.info('lane_position %s m' % lane_position)
        logger.info('lane_width %s m' % lane_width)

        debug_output[nonzero_y[left_lane_indexes], nonzero_x[left_lane_indexes]] = [255, 0, 0]
        debug_output[nonzero_y[right_lane_indexes], nonzero_x[right_lane_indexes]] = [0, 0, 255]

        # Print detected line on top of image
        draw_fit_curves_on_image(debug_output, left_fitx, right_fitx, ploty, DRAWN_CURVE_LINE_WIDTH)

    # curvature in meters
    left_curverad_m, right_curverad_m = curvatures_in_meters(
        left_x, left_y, ploty, right_x, right_y, xm_per_pix, ym_per_pix)

    if with_debug_image:
        logger.info("Curvature right: %s m, left: %s m" % (left_curverad_m, right_curverad_m))

    lane_info = LaneInfo()
    lane_info.left_fit = left_fit
    lane_info.right_fit = right_fit
    lane_info.left_fitx = left_fitx
    lane_info.right_fitx = right_fitx
    lane_info.ploty = ploty
    lane_info.left_curverad_m = left_curverad_m
    lane_info.right_curverad_m = right_curverad_m
    lane_info.lane_position = lane_position
    lane_info.lane_width = lane_width
    lane_info.min_left_y = 0
    lane_info.max_left_y = wraped_binarized.shape[0]
    lane_info.min_right_y = 0
    lane_info.max_right_y = wraped_binarized.shape[0]

    return debug_output, lane_info


def local_line_search(wraped_binarized, left_fit, right_fit, ym_per_pix=30 / 720, xm_per_pix=3.7 / 700,
                      margin=100, with_debug_image=True):
    DRAWN_CURVE_LINE_WIDTH = 4  # width of final curve in pixels

    nonzero = wraped_binarized.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_indx = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) &
                      (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))

    right_lane_indx = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) &
                       (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    left_x = nonzerox[left_lane_indx]
    left_y = nonzeroy[left_lane_indx]
    right_x = nonzerox[right_lane_indx]
    right_y = nonzeroy[right_lane_indx]

    if len(left_y) == 0 or len(right_y) == 0 or len(left_x) != len(left_y) or len(right_y) != len(right_x):
        return None, None

    # Fit a second order polynomial to each
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    # Generate x and y values for plotting
    ploty = (np.linspace(0, wraped_binarized.shape[0] - 1, wraped_binarized.shape[0])).astype(np.int)
    left_fitx = (left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]).astype(np.int)
    right_fitx = (right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]).astype(np.int)

    lane_position = get_lane_position(wraped_binarized, left_fitx, right_fitx, xm_per_pix)
    lane_width = get_lane_width(left_fitx, right_fitx, xm_per_pix)

    result = np.dstack((wraped_binarized, wraped_binarized, wraped_binarized)) * 255

    if with_debug_image:
        logger.info('lane_position %s m' % lane_position)
        logger.info('lane_width %s m' % lane_width)

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((wraped_binarized, wraped_binarized, wraped_binarized)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_indx], nonzerox[left_lane_indx]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_indx], nonzerox[right_lane_indx]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                        ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                         ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        draw_fit_curves_on_image(result, left_fitx, right_fitx, ploty, DRAWN_CURVE_LINE_WIDTH)

    # curvature in meters
    left_curverad_m, right_curverad_m = curvatures_in_meters(
        left_x, left_y, ploty, right_x, right_y, xm_per_pix, ym_per_pix)

    if with_debug_image:
        logger.info("Curvature right: %s m, left: %s m" % (left_curverad_m, right_curverad_m))

    lane_info = LaneInfo()
    lane_info.left_fit = left_fit
    lane_info.right_fit = right_fit
    lane_info.left_fitx = left_fitx
    lane_info.right_fitx = right_fitx
    lane_info.ploty = ploty
    lane_info.left_curverad_m = left_curverad_m
    lane_info.right_curverad_m = right_curverad_m
    lane_info.lane_position = lane_position
    lane_info.lane_width = lane_width
    lane_info.min_left_y = 0
    lane_info.max_left_y = wraped_binarized.shape[0]
    lane_info.min_right_y = 0
    lane_info.max_right_y = wraped_binarized.shape[0]

    return result, lane_info


def get_lane_position(wraped_binarized, left_fitx, right_fitx, xm_per_pix):
    """
    Retruns the position from the center of the lane. Positive number means that the car is on the right side
    of the lane, negative left side of the lane
    """
    lane_center = compute_midpoint(left_fitx, right_fitx)
    image_center = wraped_binarized.shape[1] / 2
    lane_position = (lane_center - image_center) * xm_per_pix
    return lane_position


def get_lane_width(left_fitx, right_fitx, xm_per_pix):
    """ Returns the lane width expressed in meters """
    a = left_fitx[Y_ARRAY_INDEX_OF_BOTTOM_ELEMENT]
    b = right_fitx[Y_ARRAY_INDEX_OF_BOTTOM_ELEMENT]
    return (b - a) * xm_per_pix


def draw_fit_curves_on_image(image, left_fitx, right_fitx, ploty, line_width):
    """ Prints detected line on top fo the image """
    for l_x, r_x, y in zip(left_fitx, right_fitx, ploty):
        half_line_width = int(line_width / 2)
        # this implementation could be better, but a try catch is needed when drawing near the edges of the matrix
        for x in range(l_x - half_line_width, l_x + half_line_width):
            try:
                image[y, x] = [0, 255, 255]
            except IndexError:
                pass
        for x in range(r_x - half_line_width, r_x + half_line_width):
            try:
                image[y, x] = [0, 255, 255]
            except IndexError:
                pass


def curvatures_in_meters(left_x, left_y, ploty, right_x, right_y, xm_per_pix, ym_per_pix):
    """ Returns the curvature in meters for the left and right lanes """
    left_fit_cr = np.polyfit(left_y * ym_per_pix, left_x * xm_per_pix, 2)
    right_fit_cr = np.polyfit(right_y * ym_per_pix, right_x * xm_per_pix, 2)
    left_curverad_m = compute_curvature(ploty, left_fit_cr)
    right_curverad_m = compute_curvature(ploty, right_fit_cr)
    return left_curverad_m, right_curverad_m


def compute_curvature(ploty, fit):
    """ Conputes the curvature of a line """
    y_eval = np.max(ploty)
    return ((1 + (2 * fit[0] * y_eval + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])


def compute_midpoint(left_fitx, right_fitx):
    """ Returns the midpoint of the lane """
    a = left_fitx[Y_ARRAY_INDEX_OF_BOTTOM_ELEMENT]
    b = right_fitx[Y_ARRAY_INDEX_OF_BOTTOM_ELEMENT]
    return a + (b - a) / 2


def lane_lines_directixes(lane_info):
    """
    Computes the parabola directrix for both curves.
    """

    def directrix(coefficients):
        a, b, c = coefficients
        return (b ** 2 - 4 * a * c + 1) / 4 * a

    return directrix(lane_info.left_fit), directrix(lane_info.right_fit)


def can_use_this_frame(current_lane, previous_lane):
    """
    Sanity checking, if this frame can be used.

    Checking that they have similar curvature
    Checking that they are separated by approximately the right distance horizontally
    Checking that they are roughly parallel
    """

    SIMILAR_CURVATURE_THRESHOLD = 1000  # in meters almost 2 km in average
    SIMILAR_LANE_DISTANCE_THRESHOLD = 0.3  # in meters
    ROUGHLY_PARALLEL_THRESHOLD = 0.0001

    # Checking that they have similar curvature
    left_curv_diff = np.abs(current_lane.left_curverad_m - previous_lane.left_curverad_m)
    right_curv_diff = np.abs(current_lane.right_curverad_m - previous_lane.right_curverad_m)

    curvature_ok = left_curv_diff <= SIMILAR_CURVATURE_THRESHOLD and right_curv_diff <= SIMILAR_CURVATURE_THRESHOLD

    # print('curvature ok', curvature_ok)

    # Checking that they are separated by approximately the right distance horizontally

    lane_width_diff = np.abs(current_lane.lane_width - previous_lane.lane_width)
    distance_ok = lane_width_diff <= SIMILAR_LANE_DISTANCE_THRESHOLD

    # print('distance ok', distance_ok)

    # Checking that they are roughly parallel

    current_left_directrix, current_right_directrix = lane_lines_directixes(current_lane)

    current_directrix_diff = np.abs(current_left_directrix - current_right_directrix)

    # print("%.10f" %current_directrix_diff)

    current_directrix_ok = current_directrix_diff <= ROUGHLY_PARALLEL_THRESHOLD
    # print('average parallelism ', left_tan_ok, right_tan_ok)

    return curvature_ok and distance_ok and current_directrix_ok


MAX_FRAMES_TO_SKIP_BEFORE_RESET = 3
skipped_frames_counter = 0


def detect_on_video_frame(wraped_binarized, with_debug_image):
    """ Will take a video frame and will apply a metod to extract lane lines """
    global last_valid_lanes
    is_first_frame = len(last_valid_lanes) == 0

    if is_first_frame:
        # the first time trying to find a lane from scratch
        detected_debug, current_lane = full_line_search(wraped_binarized, with_debug_image=with_debug_image)
        # this video frame can always be used
        last_valid_lanes.append(current_lane)
    else:
        previous_lane = last_valid_lanes[-1]
        detected_debug, current_lane = local_line_search(wraped_binarized, previous_lane.left_fit,
                                                         previous_lane.right_fit,
                                                         with_debug_image=with_debug_image,
                                                         margin=100)
        # IF fast search fails, use slow search to check for imaages
        if not current_lane:
            print('slow search, last one failed')
            detected_debug, current_lane = full_line_search(wraped_binarized, with_debug_image=with_debug_image)

        use_this_frame = can_use_this_frame(current_lane, previous_lane)

        # check to see if this frame can be used
        global skipped_frames_counter
        if use_this_frame:
            skipped_frames_counter = 0
            last_valid_lanes.append(current_lane)
        else:
            # this frame was skipped
            skipped_frames_counter += 1

        if skipped_frames_counter >= MAX_FRAMES_TO_SKIP_BEFORE_RESET:
            # Reset pipeline, starting with a new full search
            detected_debug, current_lane = full_line_search(wraped_binarized, with_debug_image=with_debug_image)
            last_valid_lanes = deque(maxlen=TOTAL_LAST_ENTRIES_TO_KEEP)
            last_valid_lanes.append(current_lane)

    if current_lane is None:
        print('Something bad happend, but why!?')
        print(detected_debug)
        print(current_lane)

    return detected_debug, current_lane


def get_averaged_left_right_lane_fits():
    left = deque()
    right = deque()

    for lane in last_valid_lanes:
        left.append(lane.left_fit)
        right.append(lane.right_fit)

    left_averaged_fit = np.array(left).mean(axis=0)
    right_averaged_fit = np.array(right).mean(axis=0)
    return left_averaged_fit, right_averaged_fit


def average_curves_and_get_lane(wraped_binarized, left_fit, right_fit, min_left_y, max_left_y, min_right_y, max_right_y,
                                ym_per_pix=30 / 720, xm_per_pix=3.7 / 700):
    """ Creates a new lane from the last prediticions """

    ploty = np.linspace(0, wraped_binarized.shape[0] - 1, wraped_binarized.shape[0]).astype(np.int)

    # Generate x and y values for plotting
    left_fitx = (left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]).astype(np.int)
    right_fitx = (right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]).astype(np.int)

    lane_position = get_lane_position(wraped_binarized, left_fitx, right_fitx, xm_per_pix)
    lane_width = get_lane_width(left_fitx, right_fitx, xm_per_pix)

    # Interpolate and rebuild data
    def eval_poly_2(fitted, x):
        a, b, c = fitted
        return a * x ** 2 + b * x + c

    left_y = np.array([x for x in range(min_left_y, max_left_y)], dtype=np.int)
    left_x = np.array([eval_poly_2(left_fit, y) for y in left_y], dtype=np.int)

    right_y = np.array([x for x in range(min_right_y, max_right_y)], dtype=np.int)
    right_x = np.array([eval_poly_2(right_fit, y) for y in right_y], dtype=np.int)

    if len(left_x) == 0 or len(left_y) == 0:
        print('WTF happend?')
        print(min_left_y, max_left_y, left_fit)
        print(min_right_y, max_right_y, right_fit)
        print(left_y)
        print(left_x)

    # curvature in meters
    left_curverad_m, right_curverad_m = curvatures_in_meters(left_x, left_y, ploty, right_x, right_y, xm_per_pix,
                                                             ym_per_pix)

    lane_info = LaneInfo()
    lane_info.left_fit = left_fit
    lane_info.right_fit = right_fit
    lane_info.left_fitx = left_fitx
    lane_info.right_fitx = right_fitx
    lane_info.ploty = ploty
    lane_info.left_curverad_m = left_curverad_m
    lane_info.right_curverad_m = right_curverad_m
    lane_info.lane_position = lane_position
    lane_info.lane_width = lane_width
    lane_info.min_left_y = min_left_y
    lane_info.max_left_y = max_left_y
    lane_info.min_right_y = min_right_y
    lane_info.max_right_y = max_right_y

    return lane_info


TOTAL_LAST_ENTRIES_TO_KEEP = 20
last_valid_lanes = deque(maxlen=TOTAL_LAST_ENTRIES_TO_KEEP)


def combined_line_detector(wraped_binarized, with_debug_image, is_video_frame, detection_history):
    """
    Used to detect lanes. There are two main approaches:
        - one for images (a full search is applied on each image)
        - one for videos (a mix between full search and local search is applied to successive frames)

    """
    if not is_video_frame:
        detected_debug, current_lane = full_line_search(wraped_binarized, with_debug_image=with_debug_image)
        return detected_debug, current_lane

    detected_debug, current_lane = detect_on_video_frame(wraped_binarized, with_debug_image)

    left_averaged_fit, right_averaged_fit = get_averaged_left_right_lane_fits()

    average_lane_info = average_curves_and_get_lane(wraped_binarized, left_averaged_fit, right_averaged_fit,
                                                    current_lane.min_left_y, current_lane.max_left_y,
                                                    current_lane.min_right_y, current_lane.max_right_y,
                                                    ym_per_pix=30 / 720, xm_per_pix=3.7 / 700)

    # no debug result is provided for videos
    return None, average_lane_info
