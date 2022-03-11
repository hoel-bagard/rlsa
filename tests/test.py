from argparse import ArgumentParser

import cv2
import numpy as np
from rlsa import rlsa

from tests.python_rlsa import python_rlsa


def show_img(img: np.ndarray, window_name: str = "Image"):
    """Displays an image until the user presses the "q" key.

    Args:
        img: The image that is to be displayed.
        window_name (str): The name of the window in which the image will be displayed.
    """
    while True:
        # Make the image full screen if it's above a given size (assume the screen isn't too small)
        if any(img.shape[:2] > np.asarray([1080, 1440])):
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.imshow(window_name, img)
        key = cv2.waitKey(10)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break


def main():
    parser = ArgumentParser(description=("Script to test the RLSA module."))
    parser.add_argument("img_path", type=str, help="Path to the image to use.")
    args = parser.parse_args()

    img_path = args.img_path

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, binary_img = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    hsv = vsv = 25
    out_img_c = rlsa(binary_img, hsv, vsv, hsv//10)
    out_img_python = python_rlsa(binary_img, hsv, vsv)

    imgs = cv2.hconcat([binary_img, out_img_c, out_img_python])
    show_img(imgs, "Binary input image & Processed image & Python processed image")

    diff = np.sum(out_img_c != out_img_python)
    assert diff == 0, f"Python and C results differ for {diff} pixels."

    print("Works fine!")


if __name__ == "__main__":
    main()
