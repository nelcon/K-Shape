import os


import pandas as pandas
from dataframe import DataFrame

from kshape_algorithm import *
import time
import csv
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import folium
from datetime import timedelta
import numpy
from datetime import datetime,timedelta
from matplotlib.font_manager import FontProperties

from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan
import mpl_toolkits.axisartist as axisartist

StartTime = datetime.now()

stationDic = {}
IdxtoStation = {}

def getdata():

    data = []
    row_data = []
    DataDir = '/Users/dengjiaying/GraduationProject/weather_data'

    if os.path.exists(DataDir) == True:

        for root, dirs, files in os.walk(DataDir):
            pass
    else:
        print("DataDir doesn't exits!")

    for file in files:
        if file[21:24] == 'RHU':

            s_data = []
            rows_data = []
            with open(DataDir + '/' + file) as fd:
                s = fd.read()
                days = s.split('\n')
                rows_data.append(days)
                row_data.append(rows_data)


                for day in days:
                    # print(day)
                    tokens = day.split(' ')
                    # print(tokens)
                    if len(tokens) == 6:
                        s_data.append(float(tokens[4]))
                data.append(s_data)

    for i,row in enumerate(row_data):
        stationId = row[0][0].split(' ')[0]
        stationDic[stationId] = i
        IdxtoStation[str(i)] = stationId
    return data,row_data




# frame = 5
# row_cnt = 0
# for row in range(len(data)):
#     row_data = data[row]
#     colcnt = 0
#     newrow_data = []
#
#     for i in range((int)(len(row_data)/frame)):
#         sum = 0
#         for j in range(frame):
#             sum += row_data[i*frame+j]
#         newrow_data.append(sum/frame)
#     sum = 0
#     if len(row_data)%frame!=0:
#         for i in range((int)(len(row_data)/frame)*frame,len(row_data)):
#             sum += row_data[i]
#         newrow_data.append(len(row_data)%frame)
#     data[row] = newrow_data
#
#
# print(data)



def get_baseline(data,w):
    data_baseline = []
    data_rasidual = []
    for i in range(len(data)):
        baseline = []
        rasidual = []
        sum = 0
        for j in range(w):
            sum += data[i][j]
        baseline.append(sum / w)
        rasidual.append(data[i][w - 1] - sum / w)

        for j in range(w, len(data[0])):
            sum = sum + data[i][j] - data[i][j - w]
            baseline.append(sum / w)
            rasidual.append(data[i][j] - sum / w)
        data_baseline.append(baseline)
        data_rasidual.append(rasidual)

    # visual plot
    # ax = plt.subplot(3, 1, 1)
    # ax.set_title("raw data")
    # plt.plot(np.arange(len(data[0])), data[0])
    #
    # ax = plt.subplot(3, 1, 2)
    # ax.set_title("baseline")
    # plt.plot(np.arange(len(data_baseline[0])), data_baseline[0])
    #
    # ax = plt.subplot(3, 1, 3)
    # ax.set_title("rasidual")
    # plt.plot(np.arange(len(data_rasidual[0])), data_rasidual[0])
    # plt.tight_layout()
    # plt.savefig('./result/baseline.png')
    return data_baseline,data_rasidual






# seed = 0
# print(data)
# distortions = []

#
# dislist = []

# for i in range(10,30):
# clusters,dis = kshape(data_baseline,27)
    # ks = KShape(n_clusters=i, verbose=True, random_state=seed)
    # y_train = ks.fit_predict(data_baseline)
    # dislist.append(dis)

# plt.plot(np.arange(10,30),dislist)
# plt.show()
# ks = KShape(n_clusters=27, verbose=True, random_state=seed)
# y_train = ks.fit_predict(data_baseline)
# print(y_pred)
# X_train = TimeSeriesScalerMinMax().fit_transform(data_baseline)
#
# print(len(set(y_train)))
# X_train = TimeSeriesScalerMinMax().fit_transform(data_baseline)
# dic = {}
# cnt = 0
# y_train = []
# for c in clusters:
#     for id in c[1]:
#         dic[id] = cnt
#     cnt += 1
# for i in range(len(data_baseline)):
#     y_train.append(dic[i])
# y_train = np.array(y_train)
# shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=X_train.shape[0],
#                                                        ts_sz=X_train.shape[1],
#                                                        n_classes=len(set(y_train)),
#                                                        l=0.1,
#                                                        r=2)
# shp_clf = ShapeletModel(n_shapelets_per_size=shapelet_sizes,
#                         optimizer=Adagrad(lr=.1),
#                         weight_regularizer=.01,
#                         max_iter=200,
#                         verbose_level=0)
# shp_clf.fit(X_train, y_train)
# predicted_locations = shp_clf.locate(X_train)
#
# print(predicted_locations)
# test_ts_id = 0
# plt.figure()
# print("begin plot")
# plt.title("Example locations of shapelet matches (%d shapelets extracted)" % sum(shapelet_sizes.values()))
# plt.plot(np.arange(len(data_baseline[test_ts_id])),data_baseline[test_ts_id])
# for idx_shp, shp in enumerate(shp_clf.shapelets_):
#     print(test_ts_id,idx_shp)
#     t0 = predicted_locations[test_ts_id, idx_shp]
#     plt.plot(numpy.arange(t0, t0 + len(shp)), shp, linewidth=2)
#
# plt.tight_layout()
# plt.show()
# print(ks.cluster_centers_)

# for i in range(1,13) :
#     ks = KShape(n_clusters=i,n_init=10,verbose=True)
#     ks.fit(data)
#     distortions.append(ks.inertia_)

# plt.plot(range(1,13), distortions, marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Distortion')
# plt.show()
# plt.plot(np.arange(len(data_baseline[0])),data_baseline[0])
# plt.show()


def getNormalize(data_baseline):

    data_baseline = TimeSeriesScalerMeanVariance().fit_transform(data_baseline)
    data_baseline = data_baseline.reshape((data_baseline.shape[0],data_baseline.shape[1]))

    return data_baseline






def plotRes(clusters,data_baseline):
    namecnt = 1

    DataDir = './result'

    if os.path.exists(DataDir) == True:

        for root, dirs, files in os.walk(DataDir):
            pass
        for file in files:
            os.remove("./result/"+file)

    for c in clusters:
        # print(len(c))
        for i in range (len(c[1])):
            idx = c[1][i]
            plt.plot(np.arange(len(data_baseline[idx])),data_baseline[idx],color='0.65',linewidth=1)
        plt.plot(np.arange(len(c[0])), c[0], color='r')
        plt.savefig('./result/' + str(namecnt) + '.png')
        # plt.show()
        namecnt += 1
        plt.clf()

def plotMap():

        China_map = folium.Map(location=[35, 120], zoom_start=4)
        clusters = []

        cnt = 0

        color = ["#FF6347", "#FF6347", "#F08080", "#FF1493", "#FF8C00", "#DA70D6", "#DAA520", "#FF00FF", "#BDB76B",
                 "#9932CC",
                 "#8A2BE2", "#FFD700", "#6A5ACD", "#4169E1", "#87CEFA", "#2E8B57", "#6495ED", "#1E90FF", "#4682B4",
                 "#87CEFA",
                 "#008B8B", "#48D1CC", "#00FA9A", "#2E8B57", "#556B2F", "#FFFF00", "#FFD700", "#FFA500", "#8B4513",
                 "#FF4500"]

        # with open('./static/TableResult/GroupResult.csv') as fd:
        with open('./result/GroupResult.csv') as fd:
            reader = csv.reader(fd)
            _station = []
            for row in reader:
                if row[0]=="ClusterId":
                    pass
                elif  row[0] == str(cnt):
                    _station.append(row)

                else:
                    clusters.append(_station)
                    _station = []
                    _station.append(row)
                    cnt += 1

        cnt = 0

        for c in clusters:
            for s in c:
                # print(s)
                folium.CircleMarker(
                    location=[float(s[5]), float(s[4])],
                    radius=6,
                    popup=s[0] + ',' + s[1] + ',' + s[2],
                    fill=True,
                    color=color[cnt],
                    fillcolor=color[cnt]
                ).add_to(China_map)
            cnt += 1

        # China_map.save('./templates/map.html')
        China_map.save('./result/map.html')

# plotRes(clusters)




def WriteResToFile(clusters,row_data):
    # 显示城市
    stations = {}
    with open('/Users/dengjiaying/GraduationProject/StationC.csv') as fd:
        reader = csv.reader(fd)
        for row in reader:
            stations[row[0]] = [row[0],row[1],row[2],row[3],row[4]]


    #写文件
    filename = "GroupResult"
    wirtefile = str(filename) + '.csv'
    # savefile = os.path.join('./static/TableResult',wirtefile)
    savefile = os.path.join('./result',wirtefile)


    with open(savefile,mode='w') as fd:
        writer = csv.writer(fd)
        writer.writerow(["ClusterId","StationId","Province","StationName","latitidue","longtitude"])

        cnt = 0
        for c in clusters :

            cstation = []
            for s in c[1]:
                id = row_data[s][0][0].split(' ')[0]
                if id in stations:
                    cstation.append(stations[id])
            cstation = sorted(cstation, key=(lambda x: [x[1], x[2]]))
            for _station in cstation:
                _station.insert(0,str(cnt))
                t= _station
                writer.writerow(t)
            cnt += 1

# WriteResToFile(clusters)

def cutdata(data,start,end):
    newdata = []
    for i in range(len(data)):
        newdata.append(data[i][start:end])
    return  newdata







def beginKShape(k,begindate,enddate,idx = None):
    data, row_data = getdata()
    print("get data done")
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

    clusters,idx = kshape(inputdata, k,idx)
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
    #
    # return SSE

# beginKShape(29,"01/01/1966","04/01/1966")
# beginKShape(29,"01/01/1967","04/01/1967")
# beginKShape(29,"01/01/1968","04/01/1968")
# beginKShape(29,"01/01/1969","04/01/1969")

def drawpic():
    fig = plt.figure()
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    # plt.plot(x,ans,marker='s',lw=2,c='black')
    plt.xlabel('time series length')
    ax.axis["bottom"].set_axisline_style("->", size = 1.5)
    ax.axis["left"].set_axisline_style("->", size = 1.5)
    #通过set_visible方法设置绘图区的顶部及右侧坐标轴隐藏
    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)
    plt.savefig('./result/kshapetime.png')




def beginDynamicKShape(k,d1,d2):
    StartTime = datetime.now()
    data, row_data = getdata()

    # begindate = time.strptime(begindate, "%m/%d/%Y")
    # enddate = time.strptime(enddate, "%m/%d/%Y")
    # firstday = time.strptime("01/01/1960", "%m/%d/%Y")
    #
    # begindate = datetime(begindate[0], begindate[1], begindate[2])
    # enddate = datetime(enddate[0], enddate[1], enddate[2])
    # firstday = datetime(firstday[0], firstday[1], firstday[2])
    #
    # print(begindate.month)
    # d1 = (begindate-firstday).days
    # d2 = (enddate-firstday).days

    # testdate = begindate + +timedelta(days=d2)
    # print(testdate.strftime("%m/%d/%Y"))

    step = 30
    stationId = 50353

    mindis = 0.09
    maxdis = 0.2
    now = d1
    end = now + step
    oldavgdis = -1

    answer = []
    oldclusters = []

    oldidx = None

    def GetAvgDis(clusters):
        totaldis = 0
        cnt = 0
        for c in clusters:
                for d in c[2]:
                    totaldis += d
                    cnt += 1
        return  totaldis/float(cnt)

    while True:

        cutnewdata = cutdata(data,now,end)
        data_baseline,data_rasidual = get_baseline(cutnewdata,w=15)
        inputdata = getNormalize(data_baseline)
        clusters , oldidx = DynamicKShape(inputdata,k,oldidx)
        avgdis = GetAvgDis(clusters)

        print("cluster done")


        print("it now end dis:",now,end,avgdis)
        if avgdis < mindis :

            end += step

        elif avgdis >= mindis and avgdis < maxdis:

            if  end - now > step and oldavgdis < mindis :
                print("now end:", now, end - step, oldavgdis)
                answer.append([now,end-step])
                now = end - step
                end = now + step
            else:
                end += step

        else:
            if end - now == step:
                print("now end null:", now, end, avgdis)
                answer.append([now,end])
                now = end
                end = now + step
            else:
                print("now end:", now, end - step, oldavgdis)
                answer.append([now,end-step])
                now = end - step
                end = now + step


        # if end > len(data[0]) :
        #     break
        if end > d2:
            break

        oldclusters = clusters
        oldavgdis = avgdis
    EndTime = datetime.now()
    print('total cost time:', (EndTime - StartTime).seconds)
    return  answer
# beginDynamicKShape(29,0,2000)
# beginDynamicKShape(25,"01/01/1960","01/01/1965")
#
# stations = {}
# with open('/Users/dengjiaying/GraduationProject/StationC.csv') as fd:
#     reader = csv.reader(fd)
#     for row in reader:
#         stations[row[0]] = [row[0], row[1], row[2], row[3], row[4]]
#
# for i in range(len(answer)):
#     print("i:",i)
#     for c in answer[i][2]:
#         if stationDic[str(stationId)] in c[1]:
#             for station in c[1]:
#                 id = IdxtoStation[str(station)]
#                 if id in stations:
#                     print(stations[id][1],stations[id][2])
#                 else:
#                     print("miss ",id)
#             break


# beginDynamicKShape(29,1680,2000)

