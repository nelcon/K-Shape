import csv
from kshape_algorithm import *
import sklearn
from tslearn.clustering import KShape

save_dir = 'city'
filename = 'AP023_afterprocess.csv'

weather_data = []
with open(save_dir+'/'+filename) as f:
    reader = csv.reader(f)
    for row in reader:
        row_data = list(row)
        weather_data.append(row_data)

# print(weather_data)
print(weather_data[0][1])

airtemp = []
humidity = []
for row in weather_data:
    row_airtemp = []
    row_humidity = []
    list_weatherdata = list(row[1])
    print(list_weatherdata)
    # for i in row[1]:
    #     print(i)
        # row_airtemp.append(i[0])
        # row_humidity.append(i[1])
    # airtemp.append(row_airtemp)
    # humidity.append(row_humidity)

# print(airtemp)