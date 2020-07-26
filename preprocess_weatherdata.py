import csv
import os
from kshape_algorithm import *
import matplotlib.pyplot as plt



station_name = "AP023"
save_dir = 'city/'
filename = save_dir+station_name+".csv"

#读取csv文件
datas = []
with open(filename) as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] == station_name:
            row_data = []
            row_data.append(row[1])                        #time
            if row[2] == '':
                row_data.append(-1)
            else:
                row_data.append(float(row[2]))
            if row[3] == '':
                row_data.append(-1)
            else:
                row_data.append(float(row[3]))
            datas.append(row_data)

#以time为key用字典记录
datas_dict = {}
for row_data in datas:
    time = row_data[0][0:10]+'-'+row_data[0][11:13]
    if time not in datas_dict.keys():
        if row_data[1] == -1 and row_data[2] == -1:
            datas_dict[time] = [0,0,0,0]
        elif row_data[1] == -1 and row_data[2] != -1:
            datas_dict[time] = [0,0,row_data[2],1]
        elif row_data[1] != -1 and row_data[2] == -1:
            datas_dict[time] = [row_data[1],1,0,0]
        else:
            datas_dict[time] = [row_data[1],1,row_data[2],1]

    else:
        if row_data[1] != -1:
            datas_dict[time] = [datas_dict[time][0]+row_data[1],datas_dict[time][1]+1,datas_dict[time][2],datas_dict[time][3]]
        if row_data[2] != -1:
            datas_dict[time] = [datas_dict[time][0],datas_dict[time][1],datas_dict[time][2]+row_data[2],datas_dict[time][3]+1]

# print(datas_dict)

#算出每个小时的平均值
for keys in datas_dict:
    if datas_dict[keys][1] == 0 and datas_dict[keys][3] == 0 :
        datas_dict[keys] = [-1,-1]
    elif datas_dict[keys][1] != 0 and datas_dict[keys][3] == 0 :
        datas_dict[keys] = [datas_dict[keys][0]/datas_dict[keys][1],-1]
    elif datas_dict[keys][1] == 0 and datas_dict[keys][3] != 0 :
        datas_dict[keys] = [-1,datas_dict[keys][2]/datas_dict[keys][3]]
    else:
        datas_dict[keys] = [datas_dict[keys][0]/datas_dict[keys][1],datas_dict[keys][2]/datas_dict[keys][3]]

# print(datas_dict)
#按时间排序
sorted(datas_dict)

#构造以天为单位，长度为24的时间序列
daydata_dict = {}

for keys in datas_dict:
    daykey = keys[0:10]
    if daykey not in daydata_dict:
        daydata_dict[daykey] = [datas_dict[keys]]
    else:
        daydata_dict[daykey].append(datas_dict[keys])

# print(daydata_dict)
for keys in list(daydata_dict):
    if len(daydata_dict[keys]) != 24 :
        # print(keys)
        del(daydata_dict[keys])


for keys in list(daydata_dict):
    for i in daydata_dict[keys]:
        if int(i[0])==-1 or int(i[1])== -1:
            del (daydata_dict[keys])
            break


#将时间序列数据写入文件
# writerfilename = station_name + '_afterprocess.csv'
#
# save_file =  os.path.join(save_dir[0:4], writerfilename)
#
# with open(save_file, mode='w') as fd:
#     writer = csv.writer(fd)
#     for keys in daydata_dict:
#         writer.writerow([keys,daydata_dict[keys]])

air_temp = []
for keys in daydata_dict:
    row_airtemp = []
    for i in daydata_dict[keys]:
        row_airtemp.append(i[0])
    air_temp.append(row_airtemp)


plt_cnt = 0

t_airtemp = []
for row in air_temp:
    # plt.plot(np.arange(len(row)),row)
    t_airtemp.append(row)
    plt_cnt += 1
    if plt_cnt > 100:
        break

clusters = kshape(air_temp,2)
print(clusters)
for c in clusters:
    plt.plot(np.arange(len(c[0])),c[0])
plt.show()


print("preprocess end")



