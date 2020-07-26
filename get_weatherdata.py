import os
import requests
import json
import time

apikey = 'KzjlYhV3d48PwbGtAVa1zMDiOaMiiIhqzqw'
save_dir = 'city'

# proxies = {"http": "http://127.0.0.1:1080", "https": "http://127.0.0.1:1080"}

r = requests.get('https://api.synopticlabs.org/v2/auth', {'apikey': apikey})
res = json.loads(r.content.decode())
token = res['TOKEN']
print('token:', token)
#73113b0884fc4c829b7a7c99c3550bf5
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# stations = ['BGGH']
stations = []
# with open('APRSWXNET_CWOP Stations.html') as fd:
#     s = fd.read()
#     print(s)
#     sf = 'mesowest.utah.edu/cgi-bin/droman/meso_base_dyn.cgi?stn='
#     se = '"'
#     idx1 = 0
#     while idx1 >= 0:
#         idx1 = s.find(sf, idx1+1)
#         idx2 = s.find(se, idx1+1)
#         if idx1 < 0:
#             break
#         stations.append(s[idx1 + len(sf): idx2])
#
# print('stations num:', len(stations))
stations.append('AP023')
for stn in stations:
    save_file = os.path.join(save_dir, '%s.csv' % stn)
    if os.path.exists(save_file):
        print('find %s, skip to next.' % save_file)
        continue
    print(stn)
    while True:
        try:
            r = requests.get('https://api.synopticlabs.org/v2/stations/timeseries', {
                'start': '199801010000',
                'end': '201902180000',
                'output': 'csv',
                'token': token,
                'stid': stn,
                'vars': 'air_temp,relative_humidity,weather_condition',
                'obtimezone':'local'
            })
            break
        except:
            print('requests failed.')
            time.sleep(3)
    with open(save_file, mode='wb') as fd:
        fd.write(r.content)

print("get_weatherdata end")
