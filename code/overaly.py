import cv2
import numpy as np


def overlay_detected_lane(wraped_binarized, undist, Minv, left_fitx, right_fitx, ploty):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(wraped_binarized).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the wraped_binarized blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (wraped_binarized.shape[1], wraped_binarized.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result


def print_curvature_and_distance(image, left_curverad_m, right_curverad_m, lane_position, lane_width):
    """ Print detections in overlay on top of the image """
    avg_curvature_km = np.average([left_curverad_m, right_curverad_m]) / 1000
    curve_radius_text = "Radius of Curvature = %.3f (km)" % avg_curvature_km

    car_position_text = '<- (left)' if lane_position < 0.0 else '-> (right)'
    position_from_center_text = "Vehicle is %.2fm %s of center" % (np.abs(lane_position), car_position_text)

    lane_width_text = "Lane width %.2fm" % lane_width

    FONT_SIZE = 1.5
    COLOR = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = cv2.putText(image, curve_radius_text, (10, 50), font, FONT_SIZE, COLOR, 2, cv2.LINE_AA)
    image = cv2.putText(image, position_from_center_text, (10, 50 * 2), font, FONT_SIZE, COLOR, 2, cv2.LINE_AA)
    image = cv2.putText(image, lane_width_text, (10, 50 * 3), font, FONT_SIZE, COLOR, 2, cv2.LINE_AA)

    return image
