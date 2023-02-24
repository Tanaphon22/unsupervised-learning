import numpy
import numpy as np

W = numpy.load("D:/OneDrive_1_2-13-2023/w.npy")
X = numpy.load("D:/OneDrive_1_2-13-2023/x.npy")
Y = numpy.load("D:/OneDrive_1_2-13-2023/y.npy")
Z = numpy.load("D:/OneDrive_1_2-13-2023/z.npy")


#unique, counts = numpy.unique(Z, return_counts=True)
#print(unique,counts)

#print(W.shape, len(W))
#print(X.shape, len(X))
#print(Y.shape, len(Y))
#print(Z.shape, len(Z))

def contingency(input1, input2):
    if len(input1) == len(input2):
        f00 = 0
        f01 = 0
        f10 = 0
        f11 = 0
        for i in range(len(input1)):
            if input1[i] == 0 and input2[i] == 0:
                f00 += 1
            elif input1[i] == 0 and input2[i] == 1:
                f01 += 1
            elif input1[i] == 1 and input2[i] == 0:
                f10 += 1
            elif input1[i] == 1 and input2[i] == 1:
                f11 += 1
        print(f11, f10)
        print(f01, f00)
        return f00, f10, f01, f11
    else:
        print("Size error")

def sym_binary_coef(input1,input2):
    f00, f01, f10, f11 = contingency(input1, input2)
    return (f00+f11)/(f00 + f01 + f10 + f11)

def similarity_matrix(list_of_input):
    n = len(list_of_input)
    output = np.zeros((n, n))
    for i in range(0, n-1):
        for j in range(i+1, n, 1):
            output[j][i] = sym_binary_coef(list_of_input[i], list_of_input[j])
    print(output)

def sym_binary_coef2(input1,input2):
    f00, f01, f10, f11 = contingency(input1, input2)
    return (f01+f10)/(f00 + f01 + f10 + f11)

def Disssimilarity_matrix(list_of_input):
    n = len(list_of_input)
    output = np.zeros((n, n))
    for i in range(0, n-1):
        for j in range(i+1, n, 1):
            output[j][i] = sym_binary_coef2(list_of_input[i], list_of_input[j])
    print(output)

#contingency(W,X)
#contingency(X,Y)
#contingency(Y,Z)

#similarity_matrix([W,X,Y,Z])
Disssimilarity_matrix([W, X, Y, Z])


