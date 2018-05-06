import logging
import os

import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip

from code.constants import OUTPUT_IMAGES_DIR
from code.pipeline import process_raw, set_debug_mode_options, reset_pipeline
from code.utils import files_in_directory, get_module_directory, create_directory_if_missing, \
    add_suffix_before_extension, fetch_image_at_time

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s][%(name)s]: %(message)s'
)

logger = logging.getLogger(__name__)


def images_pipe(images_directory_path):
    """
    Loads images from disk and pushes them to the pipeline, stores the image to the output directory
    """
    source_image_dir = os.path.join(get_module_directory(), '..', images_directory_path)
    images_to_process_full_path = files_in_directory(source_image_dir, supported_extensions=["jpg"])
    images_to_process_names = files_in_directory(source_image_dir, supported_extensions=["jpg"], full_path=False)

    output_dir_path = os.path.join(get_module_directory(), '..', OUTPUT_IMAGES_DIR)
    create_directory_if_missing(output_dir_path)

    # Images pipeline always processes in debug mode with debug output
    set_debug_mode_options(is_debug=True, use_debug_output=True)

    for image_path, image_name in zip(images_to_process_full_path, images_to_process_names):
        # load image
        img = cv2.imread(image_path)
        print(image_name)

        # run pipeline and get output
        processed_output = process_raw(img, is_video_frame=False)

        # save processed image results
        for processed_image, label in processed_output:
            output_image_name = add_suffix_before_extension(image_name, label)

            output_file_path = os.path.join(output_dir_path, output_image_name)
            cv2.imwrite(output_file_path, processed_image)


def video_pipe(video_name):
    """
    Loads a video and pushes each frame to the pipeline,
    stores each frame in a new video in th output directory
    """
    in_video_path = os.path.join(get_module_directory(), '..', video_name)
    out_video_path = os.path.join(get_module_directory(), '..', OUTPUT_IMAGES_DIR, video_name)
    clip3 = VideoFileClip(in_video_path)

    # disable logging during video processing
    reset_pipeline()
    set_debug_mode_options(is_debug=False, use_debug_output=False)
    challenge_clip = clip3.fl_image(process_raw)
    challenge_clip.write_videofile(out_video_path, audio=False)

    # enable logging after video processing is over
    set_debug_mode_options(is_debug=True, use_debug_output=False)


def main():
    # images_pipe(TEST_IMAGES_DIR)
    video_pipe('project_video.mp4')
    video_pipe('challenge_video.mp4')  # disabled, error on this video
    video_pipe('harder_challenge_video.mp4')

    # Extract frames to enhance the piepline
    #fetch_image_at_time('project_video.mp4', time_ms=41500)


if __name__ == '__main__':
    main()
