import cv2
import numpy as np
from skimage import exposure
from skimage.feature import hog


def resize_and_gray(image_bgr, size):
    resized = cv2.resize(image_bgr, size)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return resized, gray


def extract_hog(gray, visualize=False):
    feature, hog_img = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        visualize=visualize,
        feature_vector=True,
    )

    if not visualize:
        return feature, None

    hog_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 10))
    hog_uint8 = (hog_rescaled * 255).astype(np.uint8)
    hog_bgr = cv2.cvtColor(hog_uint8, cv2.COLOR_GRAY2BGR)
    return feature, hog_bgr
