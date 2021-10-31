# https://blog.csdn.net/dangsh_/article/details/81084221
import requests
from lxml import etree
import matplotlib.pyplot as plt
from pandas import Series

url = "http://datachart.500.com/ssq/history/newinc/history.php?start=00001&end=18081"
response = requests.get(url)
response = response.text
selector = etree.HTML(response)
reds = []
blues = []
for i in selector.xpath('//tr[@class="t_tr1"]'):
    datetime = i.xpath('td/text()')[0]
    red = i.xpath('td/text()')[1:7]
    blue = i.xpath('td/text()')[7]
    for i in red:
        reds.append(i)
    blues.append(blue)
