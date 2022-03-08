import numpy as np
from rlsa import rlsa


def main():
    print("Start")

    inputs = np.linspace(0, 1, 5)
    outputs = rlsa(inputs)
    print(f"{inputs=}, {outputs=}")


if __name__ == "__main__":
    main()
