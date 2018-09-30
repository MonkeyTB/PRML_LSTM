#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-
import numpy as p
from numpy import *
import matplotlib.pyplot as plt
import time as t
data = []
# 高斯分布获取点和绘图
def guess(point,voc,color):
    x,y = p.random.multivariate_normal(point,voc,1000).T
    for i in range(len(x)):
        data.append([x[i],y[i]])
    plt.plot(x,y,color)
# 初始中心点
point = [[0.2,0.7],[0.8,0.4],[0.1,0.2]]
# 颜色
mark = ['ro','go','bo']
voc = [[0.1,0],[0,0.1]]
for i in range(3):
    guess(point[i],voc,mark[i])
plt.show()