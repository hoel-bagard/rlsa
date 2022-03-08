from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from rlsa import rlsa


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
    parser.add_argument("img_path", type=Path, help="Path to the image to use.")
    args = parser.parse_args()

    img_path: Path = args.img_path

    img = cv2.imread(str(img_path), 0)

    out_img = rlsa(img, 35, 35)

    out_img = out_img.astype(np.uint8)

    show_img(img, "Input image")
    show_img(out_img, "Processed image")


if __name__ == "__main__":
    main()
