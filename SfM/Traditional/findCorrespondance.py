""" File to implement function to find correspondence between 2 images
"""
import numpy as np


def findCorrespondance(a, b):
    """To extract corrospondance between 2 images from given file

    Args:
        a (int): First Image Number
        b (int): Second Image Number

    Returns:
        list: List of matching points: R,G,B,x1,y1,x2,y2
    """
    database_path = "Data/"
    matching_list = []

    if (1 <= a <= 6):
        with open(database_path + "matching" + str(a) + ".txt") as f:
            line_no = 1
            for line in f:
                if line_no == 1:
                    line_no += 1
                    nfeatures = line[11:15]
                    nfeatures = int(nfeatures)

                else:
                    matching_list.append(line.rstrip('\n'))

    else:
        print("First image argument Number not found")

    final_list = []
    for i in range(0, len(matching_list)):
        current_row = matching_list[i]
        splitStr = current_row.split()
        current_row = []
        for j in splitStr:
            current_row.append(float(j))
        final_list.append(np.transpose(current_row))

    rgb_list = []
    x_list = []
    y_list = []
    binary_list = []
    for i in range(0, len(final_list)):
        rgb_row = []
        x_row = []
        y_row = []
        binary_row = []
        current_row = final_list[i]
        current_row = current_row[1:len(current_row)]

        res = np.where(current_row == b)

        x_row.append(current_row[3])
        y_row.append(current_row[4])
        binary_row.append(1)

        if (len(res[0]) != 0):
            index = res[0][0]

            rgb_row.append(current_row[0])
            rgb_row.append(current_row[1])
            rgb_row.append(current_row[2])
            x_row.append(current_row[index + 1])
            y_row.append(current_row[index + 2])
            binary_row.append(1)

        else:
            x_row.append(0)
            y_row.append(0)
            binary_row.append(0)

        if (len(x_row) != 0):
            x_list.append(np.transpose(x_row))
            y_list.append(np.transpose(y_row))
            binary_list.append(np.transpose(binary_row))
            rgb_list.append(np.transpose(rgb_row))

    return np.array(x_list), np.array(y_list), np.array(binary_list), np.array(
        rgb_list)
