import numpy as np
from rlsa import rlsa


def main():
    print("Start")

    inputs = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    outputs = rlsa(inputs, 3, 4)
    print(f"Inputs:\n{inputs}\nOutputs:\n{outputs}")


if __name__ == "__main__":
    main()
