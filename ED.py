
import math
import numpy as np

from numpy.random import randint, seed
from numpy.linalg import norm, eigh
from numpy.linalg import norm
from numpy.fft import fft, ifft
from mainFile import *
from numpy.random import seed; seed(1)
from datetime import datetime,timedelta
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def zscore(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    mns = a.mean(axis=axis)
    sstd = a.std(axis=axis, ddof=ddof)
    if axis and mns.ndim < a.ndim:
        res = (((a - np.expand_dims(mns, axis=axis)) /
                np.expand_dims(sstd, axis=axis)))
    else:
        res = (a - mns) / sstd
    return np.nan_to_num(res)






def _extract_shape(idx, x, j):

    a = []
    for i in range(len(idx)):
        if idx[i] == j:
            a.append(x[i])
    if len(a) == 0:
        a = np.zeros((1,x.shape[1]))
    else:
        a = np.array(a)
        a = a.mean(0)


    return a

def ED(a,b):
    dis = 0
    if len(a)!=len(b):
        print("ED != !!!!")
    for i in range(0,len(a)):
        dis += abs(a[i]-b[i])
    return dis


def dtw_distance(dataset1, dataset2):
    ''' dynamic time warping '''
    dtw = {}
    for i in range(len(dataset1)):
        dtw[(i, -1)] = float('inf')
    for i in range(len(dataset2)):
        dtw[(-1, i)] = float('inf')

    dtw[(-1, -1)] = 0

    for i in range(len(dataset1)):
        for j in range(len(dataset2)):
            dist = (dataset1[i] - dataset2[j])**2
            dtw[(i, j)] = dist + min(dtw[(i - 1, j)], dtw[(i, j - 1)], dtw[(i - 1, j - 1)])

    return math.sqrt(dtw[len(dataset1) - 1, len(dataset2) - 1])

def _kmeans(x, k, initial_clustering=None):
    """
    >>> from numpy.random import seed; seed(0)
    # >>> _kshape(np.array([[1,2,3,4], [0,1,2,3], [-1,1,-1,1], [1,2,2,3]]), 2)
    # (array([0, 0, 1, 0]), array([[-1.2244258 , -0.35015476,  0.52411628,  1.05046429],
    #        [-0.8660254 ,  0.8660254 , -0.8660254 ,  0.8660254 ]]))
    # >>> _kshape(np.array([[1,2,3,4], [0,1,2,3], [0,1,2,3], [1,2,2,3]]), 2)
    (

    )
    """
    m = x.shape[0]

    if initial_clustering is not None:
        assert len(initial_clustering) == m, "Initial assigment does not match column length"
        idx = initial_clustering
    else:
        idx = randint(0, k, size=m)


    centroids = np.zeros((k, x.shape[1]))
    distances = np.empty((m, k))
    dis = np.zeros(m)

    for _ in range(50):
        old_idx = idx
        print(_," start:")
        for j in range(k):
            centroids[j] = _extract_shape(idx, x,j)

        for i in range(m):
            # print("i:",i)
            for j in range(k):
                distances[i, j] = dtw_distance(x[i],centroids[j])
        idx = distances.argmin(1)

        if np.array_equal(old_idx, idx):
            break

    for i in range(len(idx)):
        dis[i] = distances[i][idx[i]]

    return idx, centroids,dis


def kmeans(x, k, initial_clustering=None):
    idx, centroids,dis = _kmeans(np.array(x), k, initial_clustering)
    clusters = []
    for i, centroid in enumerate(centroids):
        series = []
        seriesdis = []
        for j, val in enumerate(idx):
            if i == val:
                series.append(j)
                seriesdis.append(dis[j])
        clusters.append((centroid, series,seriesdis))

    return clusters,idx





def silhouette_score(clusters,inputdata):
    t_cluster = []
    for c in clusters:
        if len(c[1])!=0:
            t_cluster.append(c)
    clusters = t_cluster
    distance = np.zeros((len(inputdata),len(inputdata)))

    for i in range(0,len(inputdata)):
        print("s i:",i)
        for j in range(0,len(inputdata)):
            if i == j:
                continue;
            elif distance[j][i] != 0:
                distance[i][j] = distance[j][i]
            else:
                distance[i][j] =dtw_distance(inputdata[i],inputdata[j])


    answer = []
    for ida,c in enumerate(clusters):
        for s in c[1]:
            # print("one begin")
            a = 0
            b = float("inf")
            for sn in c[1]:
                if s == sn:
                    continue
                a += distance[s][sn]
            if len(c[1])==1 :
                a = 0
            else:
                a = float(a) / (len(c[1]) - 1)

            # t_distance = []
            # print("b:")
            for idb,cn in enumerate(clusters):

                if ida == idb:
                    continue
                tb = 0
                for sn in cn[1]:
                    tb += distance[s][sn]
                tb = tb / float(len(cn[1]))
                # if tb < b :
                #     t_distance = []
                #     for sn in cn[1]:
                #         t_distance.append(distance[s][sn])
                b = min(b,tb)
            # for t in t_distance:
            #     print(t)
            # print(a,b)
            s = (b-a)/max(a,b)
            # print("one end")
            answer.append(s)


    return answer
def plotRes(clusters,data_baseline):
    namecnt = 1


    for c in clusters:
        # print(len(c))
        for i in range (len(c[1])):
            idx = c[1][i]
            plt.plot(np.arange(len(data_baseline[idx])),data_baseline[idx],color='0.65',linewidth=1)
        plt.plot(np.arange(len(c[0])), c[0], color='r')
        plt.savefig('./EDResult/' + str(namecnt) + 'kmeans.png')
        # plt.show()
        namecnt += 1
        plt.clf()

def beginKSMeans(k,begindate,enddate):
    data, row_data = getdata()
    # print("get data done")
    begindate = time.strptime(begindate, "%m/%d/%Y")
    enddate = time.strptime(enddate, "%m/%d/%Y")
    firstday = time.strptime("01/01/1960", "%m/%d/%Y")

    begindate = datetime(begindate[0], begindate[1], begindate[2])
    enddate = datetime(enddate[0], enddate[1], enddate[2])
    firstday = datetime(firstday[0], firstday[1], firstday[2])

    d1 = (begindate-firstday).days
    d2 = (enddate-firstday).days
    print(d1,d2)
    print(d2-d1)
    cutnewdata = cutdata(data,d1,d2)

    # baseline and rasidual
    data_baseline,data_rasidual = get_baseline(cutnewdata,w=15)
    # data_baseline = cutnewdata
    inputdata = getNormalize(data_baseline)
    StartTime = datetime.now()

    clusters,idx = kmeans(inputdata, k)
    EndTime = datetime.now()
    print("length:",d2-d1)

    # ans.append((EndTime - StartTime).seconds)
    #
    SSE = 0
    for c in clusters:
        for dis in c[2]:
            SSE += dis
    answer = silhouette_score(clusters,inputdata)

    ans = sum(answer)/len(answer)
    print("ans:",ans)
    print("SSE:",SSE)
    print('total cost time:', (EndTime - StartTime).seconds)

    # plotRes(clusters, inputdata)
    # WriteResToFile(clusters,row_data)
    # plotMap()
    return idx

beginKSMeans(29,"01/01/1966","04/01/1966")
beginKSMeans(29,"01/01/1967","04/01/1967")
beginKSMeans(29,"01/01/1968","04/01/1968")
beginKSMeans(29,"01/01/1969","04/01/1969")


# if __name__ == "__main__":
#     import doctest

