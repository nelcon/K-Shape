import csv


stations = {}

with open('/Users/dengjiaying/GraduationProject/stationpos.csv') as fd:
    reader = csv.reader(fd)
    for row in reader:
        stations[row[0]] = [row[0],row[1],row[2],row[3],row[4]]

with open('/Users/dengjiaying/GraduationProject/StationC.csv',mode='w') as fd:
    writer = csv.writer(fd)
    for _station in stations:
        writer.writerow(stations[_station])