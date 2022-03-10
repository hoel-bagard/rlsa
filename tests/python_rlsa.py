from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt


def rlsa_horizontal(img: npt.NDArray[np.uint8], value: int) -> npt.NDArray[np.uint8]:
    """Apply the RLS algorithm horizontally on the given image.

    Note: This function can be used to do the operation vertically by simply passing the transpose.

    This function eliminates horizontal white runs whose lengths are smaller than the given value.

    Args:
        img (np.ndarray): The binary image to process.
        value (int): The treshold smoothing value (hsv in the paper).

    Returns:
        The resulting image/mask.
    """
    img = img.copy()
    rows, cols = img.shape
    for row in range(rows):
        count = 0  # Index of the last 0 found
        for col in range(cols):
            if img[row, col] == 0:
                if (col-count) <= value and value != 0:
                    img[row, count:col] = 0
                count = col
    return img


def python_rlsa(img: npt.NDArray[np.uint8],
                value_horizontal: int,
                value_vertical: int,
                ahsv: Optional[int] = None) -> npt.NDArray[np.uint8]:
    """Run Length Smoothing Algorithm.

    Args:
        img (np.ndarray): The image to process.
        value_horizontal (int): The horizontal threshold (hsv=300 in the paper)
        value_vertical (int): The vertical threshold (vsv=500 in the paper)

    Returns:
        The resulting image.
    """
    horizontal_rlsa = rlsa_horizontal(img, value_horizontal)
    vertical_rlsa = rlsa_horizontal(img.T, value_vertical).T
    combined_result = cv2.bitwise_and(horizontal_rlsa, vertical_rlsa)
    rlsa_result = rlsa_horizontal(combined_result, ahsv if ahsv else value_horizontal // 10)
    return rlsa_result
