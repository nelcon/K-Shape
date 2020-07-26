import folium
import csv
syear = 1960
for year in range(syear,syear+10):

    China_map = folium.Map(location=[35,120], zoom_start=4)
    clusters = []

    cnt = 0

    color = ["#990033",	"#FF66FF",	"#660099",	"#FFFFCC",	"#99FFFF",	"#CC6699",	"#666FF",	"#FFCC00",	"#33CCCC","##FF6699",
             "#CC00FF", "#000CC" ,"#CC9000" ,"#00CC99","#FF3366" ,"#FF33FF","#6699FF",	"#CC3300","#009933","#993366",
             "#CC99FF","#666699","#3399FF","#FF3333","#99CC00","#333366","#FFFF33","#33FF99","#336666","#000066"]

    with open('/Users/dengjiaying/GraduationProject/K-Shape/city/'+str(year)+'.csv') as fd:
        reader = csv.reader(fd)
        _station = []
        for row in reader:
            if(row[0] == str(cnt)):
                if (row[0] != '0'):
                    clusters.append(_station)
                _station = []
                cnt += 1
            else:
                _station.append(row)


    cnt = 0


    for c in clusters:
        for s in c:
            print(float(s[3]),float(s[4]))
            folium.CircleMarker(
                location=[float(s[4]),float(s[3])],
                radius=6,
                popup=s[0]+','+s[1]+','+s[2],
                fill = True,
                color= color[cnt],
                fillcolor= color[cnt]
            ).add_to(China_map)
        cnt += 1

    China_map.save('/Users/dengjiaying/GraduationProject/K-Shape/map/'+str(year)+'.html')