#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
from numpy import * #python 中矩阵处理函数
from pylab import *
import function as RFTL
from sklearn.metrics import auc

def PlotRoc(Predict_array,test_array,TPR_Data,FPR_Data):
	Predict_array,test_array  = asarray(Predict_array),asarray(test_array)
	for i in range(test_array.shape[0]):#test_array.shape[0]
		sort_Data = sorted(enumerate(Predict_array[i]),key = lambda x:x[1],reverse =  True)
		total_TPR = test_array[i].tolist().count(1)
		if total_TPR == 0:
			continue
		total_FPR = test_array[i].tolist().count(0) + test_array[i].tolist().count(-1)
		#print(i,sort_Data)
		TPR, FPR = [], []
		TPR_num,FPR_num = 0.0,0.0
		for j in range(test_array.shape[1]):#test_array.shape[1]
			#print(total_TPR + total_FPR)
			if total_TPR != 0 and test_array[i][ sort_Data[j][0] ] == 1:
				TPR_num += 1/total_TPR
				TPR.append(TPR_num)
				FPR.append(FPR_num)
			else:
				FPR_num += 1/total_FPR
				TPR.append(TPR_num)
				FPR.append(FPR_num)

		TPR_Data.append(TPR)
		FPR_Data.append(FPR)


	return TPR_Data,FPR_Data
x_FPR,y_TPR = [],[]
predict_list = RFTL.readFileToList('..\data\prediction100.txt',0)
test_list = RFTL.readFileToList('..\data\第一份测试数据.txt',1)
TPR,FPR = PlotRoc(np.array(predict_list),np.array(test_list),[],[])
x_FPR.append(np.array(FPR).sum(axis=0) / np.array(FPR).shape[0])
y_TPR.append(np.array(TPR).sum(axis=0) / np.array(TPR).shape[0])
print('ROC_AUC:', auc(x_FPR[0], y_TPR[0]))
plot(x_FPR[0], y_TPR[0], "r")
show()