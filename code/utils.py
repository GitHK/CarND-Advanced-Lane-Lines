import os
import pickle

import cv2
import matplotlib.pyplot as plt


def show_image(image, window_title=None):
    """ Provide an image and creates a new figure and plots color or greyscale images """
    plt.figure()
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)

    if window_title is not None:
        fig = plt.gcf()
        fig.canvas.set_window_title(window_title)
    plt.show()


def draw_images_with_title_to(img1, img2, title1, title2):
    """ Draws two images side by side and to each image a title will be drawn on top """
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img1)
    ax1.set_title(title1)
    ax2.imshow(img2)
    ax2.set_title(title2)


def show_images_with_title(img1, img2, title1, title2):
    """ Displays drawn images side by side """
    draw_images_with_title_to(img1, img2, title1, title2)
    plt.show()


def save_images_with_title(img1, img2, title1, title2, full_out_path):
    """ Saves drawn images side by side """
    draw_images_with_title_to(img1, img2, title1, title2)
    plt.savefig(full_out_path)


def bgr_to_rgb(image):
    """ Convert OpenCV read image to RGB """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def binary_image_to_gray(image):
    """ Used to convert a binarized image to gray scale """
    return image * 255


def undistort_image(img, mtx, dist):
    """ Undistort an image with provided camera calibration parameters """
    return cv2.undistort(img, mtx, dist, None, mtx)


def get_module_directory():
    """ Returns the path of the current module's directory """
    return os.path.dirname(os.path.abspath(__file__))


def files_in_directory(dir_path, supported_extensions=None, full_path=True):
    """
    list all files in the directory with their full name
    :param dir_path: directory path to list
    :param supported_extensions: list of file extensions to include in final file list
    :param full_path: the enitre path of the file will be return if True else only the file name
    :return: list of all files in directory filtered by supported extension
    """
    supported_extensions = [] if supported_extensions is None else supported_extensions

    def ends_with(file_name):
        for extension in supported_extensions:
            if not file_name.endswith(extension):
                return False
        return True

    if full_path:
        return [os.path.join(dir_path, f) for f in os.listdir(dir_path) if
                os.path.isfile(os.path.join(dir_path, f)) and ends_with(f)]
    else:
        return [f for f in os.listdir(dir_path) if
                os.path.isfile(os.path.join(dir_path, f)) and ends_with(f)]


def dump_pickle_to_file(content, file_name):
    """ Stores a pickled object to a file """
    with open(file_name, 'wb') as handle:
        pickle.dump(content, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle_from_file(file_name):
    """ Loads a pickled object from a file """
    if not os.path.isfile(file_name):
        return None
    with open(file_name, 'rb') as handle:
        return pickle.load(handle)


def create_directory_if_missing(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def file_exist_in_path(file_path):
    return os.path.isfile(file_path)


def add_suffix_before_extension(file_name_with_extension, prefix):
    """ Adds provided suffix before extension to file name """
    file_name, extension = file_name_with_extension.split('.')
    return "%s_%s.%s" % (file_name, prefix, extension)
