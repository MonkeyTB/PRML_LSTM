#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
from numpy import * #python 中矩阵处理函数
from pylab import *
import function as RFTL
from sklearn.metrics import auc


def PlotRoc(Predict_array,train_array,test_array,TPR_Data,FPR_Data,PRE_Data):
	'''
	:param Predict_array:预测结果 
	:param train_array:训练数组
	:param test_array: 测试数组
	:param TPR_Data: 返回的结果
	:param FPR_Data: 返回的结果
	:param PRE_Data: 返回的结果
	:return: TPR、FPR、PRE
	'''
	Predict_array,test_array  = asarray(Predict_array),asarray(test_array)
	#将预测结果中，训练集中1的位置将其概率置0
	for i in range(Predict_array.shape[0]):
		for j in range(Predict_array.shape[1]):
			if train_array[i][j] == 1:
				Predict_array[i][j] = 0
	#统计Prediction中每行0的个数
	zero_list = []
	for i in range(Predict_array.shape[0]):
		zero_list.append( Predict_array[i].tolist().count(0) )

	maxValue = zero_list[ zero_list.index(max(zero_list)) ]		#zero_list.index(max(zero_list))返回列表最大元素的ID

	for i in range(test_array.shape[0]):#test_array.shape[0]
		# enumerate会将数组或列表组成一个索引序列
		sort_Data = sorted(enumerate(Predict_array[i]),key = lambda x:x[1],reverse =  True)

		total_TPR = test_array[i].tolist().count(1)  	# 统计1的个数
		if total_TPR == 0:

			continue
		total_FPR = test_array[i].tolist().count(0)
		#print(i,sort_Data)
		TPR, FPR ,PRE = [], [],[]
		TPR_num,FPR_num = 0.0,0.0
		ipre_a,ipre_b = 0,0		# a 是分子，b  分母
		for j in range(test_array.shape[1]-maxValue):		#test_array.shape[1]
			# {先计算当前行的除1之外的个数  /  最终只要一行元素的个数（311/296）} * 当前第几个元素
			k = round( float( (test_array.shape[1]-zero_list[i]) / (test_array.shape[1] - maxValue) )*j )

			ipre_b += 1
			#print(total_TPR + total_FPR)
			if total_TPR != 0 and test_array[i][ sort_Data[k][0] ] == 1:
				TPR_num += 1/total_TPR
				TPR.append(TPR_num)
				FPR.append(FPR_num)
				ipre_a += 1
				PRE.append(ipre_a / ipre_b)
			else:
				FPR_num += 1/total_FPR
				TPR.append(TPR_num)
				FPR.append(FPR_num)
				PRE.append(ipre_a / ipre_b)

		TPR_Data.append(TPR)
		FPR_Data.append(FPR)
		PRE_Data.append(PRE)


	return TPR_Data,FPR_Data,PRE_Data
x_FPR,y_TPR,y_PRE = [],[],[]
predict_list = RFTL.readFileToList('..\data\prediction.txt',0)
train_list = RFTL.readFileToList('..\data\第一份训练数据.txt',1)
test_list = RFTL.readFileToList('..\data\第一份测试数据.txt',1)

TPR,FPR,PRE = PlotRoc(np.array(predict_list),np.array(train_list),np.array(test_list),[],[],[])
x_FPR.append(np.array(FPR).sum(axis=0) / np.array(FPR).shape[0])
y_TPR.append(np.array(TPR).sum(axis=0) / np.array(TPR).shape[0])
y_PRE.append(np.array(PRE).sum(axis=0) / np.array(PRE).shape[0])
print('ROC_AUC:', auc(x_FPR[0], y_TPR[0]))
print('PR_AUC:',  auc(y_TPR[0],y_PRE[0]))
plot(x_FPR[0], y_TPR[0], "r")
plot(y_TPR[0],y_PRE[0],"b")
show()

