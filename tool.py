import os
from kshape_algorithm import *
import time
import datetime
import csv


StartTime = datetime.datetime.now()

stations = []
with open('/Users/dengjiaying/GraduationProject/K-Shape/result/GroupResult.csv') as fd:
    reader = csv.reader(fd)
    for row in reader:
        stations.append([row[0],row[2]+row[3]])

newstation = []
for i,row in enumerate(stations):
    if i%4 == 0 and (i+3)<len(stations):
        newstation.append([stations[i][0],stations[i][1],stations[i+1][0],stations[i+1][1],stations[i+2][0],stations[i+2][1]
                          ,stations[i+3][0],stations[i+3][1]])
    elif i%4 == 0 and (i+3) >= len(stations):
        if (i==len(stations)-1):
            newstation.append([stations[i][0], stations[i][1]])
        if ((i+1)==len(stations)-1):
            newstation.append([stations[i][0], stations[i][1],stations[i+1][0], stations[i+1][1]])
        if ((i+2)==len(stations)-1):
            newstation.append([stations[i][0], stations[i][1],stations[i+1][0], stations[i+1][1],stations[i+2][0], stations[i+2][1]])

filename = "GroupResult2"
wirtefile = str(filename) + '.csv'
# savefile = os.path.join('./static/TableResult',wirtefile)
savefile = os.path.join('./result',wirtefile)


with open(savefile, mode='w') as fd:
    writer = csv.writer(fd)
    for row in newstation:
        writer.writerow(row)
