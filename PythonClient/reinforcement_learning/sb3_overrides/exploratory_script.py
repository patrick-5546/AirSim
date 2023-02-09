import numpy as np


def main():
    print_evaluations()


def print_evaluations(filename='evaluations.npz'):
    evaluations = np.load(filename)
    evaluations.files
    for item in evaluations.files:
        print(item, evaluations[item], sep='\n', end='\n\n')


if __name__ == '__main__':
    main()