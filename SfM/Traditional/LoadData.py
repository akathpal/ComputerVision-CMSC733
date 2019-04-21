import numpy as np

def loadData(path = "Data/"):
    n_images = 6
    Mx = []
    My = []
    M =[]
    for i in range(1, n_images):
        mx = []
        my = []
        m =[]
        for j in range(i+1, n_images + 1):

            x_list,y_list,binary_list,rgb_list = findCorrespondance(i, j,path)

            if(j==i+1):
                mx = x_list
                my = y_list
                m = binary_list
            else:
                mx = np.hstack((mx,x_list[:,1].reshape((-1,1))))
                my = np.hstack((my,y_list[:,1].reshape((-1,1))))
                m = np.hstack((m,binary_list[:,1].reshape((-1,1))))
             
        if(i==1):
            Mx = mx
            My = my
            M = m
        else:   
            
            mx = np.hstack((np.zeros((mx.shape[0],i-1)),mx))
            my = np.hstack((np.zeros((my.shape[0],i-1)),my))
            m = np.hstack((np.zeros((m.shape[0],i-1)),m))
            assert Mx.shape[1]==mx.shape[1],"SHape not matched"
            Mx = np.vstack((Mx,mx))
            assert My.shape[1]==my.shape[1],"Shape My not matched"
            My = np.vstack((My,my))
            M = np.vstack((M,m))
    return Mx,My,M


def findCorrespondance(a, b,database_path):
    """To extract corrospondance between 2 images from given file

    Args:
        a (int): First Image Number
        b (int): Second Image Number

    Returns:
        list: List of matching points: R,G,B,x1,y1,x2,y2
    """
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

        if(len(res[0]) != 0):
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

    return np.array(x_list), np.array(y_list), np.array(binary_list), np.array(rgb_list)
