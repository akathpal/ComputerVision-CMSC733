""" File to return Visibility matrix
"""
import sys

sys.dont_write_bytecode = True


def BuildVisibilityMatrix(Visibility, r_indx, print_enable=False):
    """To return Visibility matrix element

    Args:
        Visibility (array): Visibility Matrix
        r_indx (images): Index of images
        print_enable (bool, optional): To print the returning element

    Returns:
        TYPE: Element of matrix
    """
    if (print_enable):
        print(Visibility[:, r_indx])

    return Visibility[:, r_indx]
