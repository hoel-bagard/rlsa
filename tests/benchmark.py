import time
from argparse import ArgumentParser

import cv2
from rlsa import rlsa

from tests.python_rlsa import python_rlsa


def main():
    parser = ArgumentParser(description=("Script to benchmark the RLSA module."
                                         "Run with 'python -m tests.benchmark assets/rlsa_test_image.jpg'"))
    parser.add_argument("img_path", type=str, help="Path to the image to use.")
    args = parser.parse_args()

    img_path: str = args.img_path

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, binary_img = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    hsv = vsv = 25

    nb_iterations = 10  # Python is super slow, so keep that one low.
    start_time = time.perf_counter()  # Could use timeit, but I'm lazy.
    for _ in range(nb_iterations):
        _ = rlsa(binary_img, hsv, vsv, hsv//10)
    end_time = time.perf_counter()
    c_time = (end_time - start_time) / nb_iterations

    print("Starting python loop...", end="\r")
    start_time = time.perf_counter()
    for _ in range(nb_iterations):
        _ = python_rlsa(binary_img, hsv, vsv)
    end_time = time.perf_counter()
    python_time = (end_time - start_time) / nb_iterations

    print(f"C version took {c_time:.4f}s per iteration.")
    print(f"Python version took {python_time:.4f}s per iteration.")
    print(f"A speed up of {int(python_time/c_time)} times.")


if __name__ == "__main__":
    main()
