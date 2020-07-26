import numpy as np
from kshape_algorithm import *
import matplotlib.pyplot as plt

def readucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

# flist = ['UCR']
flist = ['ECGFiveDays']
if __name__ == "__main__":
    for each in flist:
        fname = each
        x_train,y_train = readucr('UCR/'+fname+'_TRAIN')
        x_test, y_test = readucr( 'UCR/' + fname + '_TEST')


# Adiac Test
#         for i in range(int(x_train.shape[0]/2)):
#             for j in range(x_train.shape[1]):
#                 x_train[i][j] = -x_train[i][j]
#
#         clusters = kshape(x_train,2)
#
#         for i in range(len(clusters[0][1])):
#             plt.plot(np.arange(x_train.shape[1]), x_train[clusters[0][1][i],:], color='r')
#         for i in range(len(clusters[1][1])):
#             plt.plot(np.arange(x_train.shape[1]), x_train[clusters[1][1][i], :], color='b')
#         plt.show()

#ECGFiveDays Test

    # for i in range(x_test.shape[0]):
    #     plt.plot(np.arange(x_test.shape[1]),x_test[i,:],color='b')
    clusters = kshape(x_train, 2)
    # print(clusters[0][0])
    plt.plot(np.arange(x_train.shape[1]), clusters[0][0], color='r')
    plt.plot(np.arange(x_train.shape[1]), clusters[1][0], color='b')
    # for i in range(len(clusters[0][1])):
    #     plt.plot(np.arange(x_train.shape[1]), x_train[clusters[0][1][i],:], color='r')
    # for i in range(len(clusters[1][1])):
    #     plt.plot(np.arange(x_train.shape[1]), x_train[clusters[1][1][i], :], color='b')
    plt.show()
