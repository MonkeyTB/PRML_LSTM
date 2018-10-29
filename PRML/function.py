#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-
import numpy as np
from numpy import *
from __init__ import *
import numpy.linalg as la
import copy

def readFileToList (FilePath,Type):
	'''
	:param FilePath:读取文件路径 
	:param Type: 1：按int类型读，0：按float类型读
	:return: 读取文件内容的list
	'''
	try:
		f = open(FilePath,"r")
		line = f.readline()
		data_list = []
		while line:
			if Type == 1:
				num = list(map(int ,line.split()))
			else:
				num = list(map(float, line.split()))
			data_list.append(num)
			line = f.readline()
		f.close()
		return data_list
	except IOError:
		print('File is not found')
		return []
def changeArray(A,VPT):
	'''
	:param A: 待改变的数组
	:param VPT: 阈值，大于阈值的元素保留
	:return: 改变后的数组
	'''
	for i in range(A.shape[0]):
		for j in range(A.shape[1]):
			if A[i][j] < VPT:
				A[i][j] = 0
	return A
def FindStepPath(RR,DD,DR):
	list = []
	#r-r-d
	for i in range(RR.shape[0]):
		for j in range(RR.shape[1]):#i
			if RR[i][j] != 0 and i != j:
				for k in range(DR.shape[1]):
					if DR[j][k] != 0:
						list.append([0,i,j,k,0])
	# np.savetxt('..\data\\rrd.txt',
	# 		   np.array(list), fmt=['%s'] * np.array(list).shape[1], newline='\n')
	#r-d-d
	for i in range(DR.shape[0]):
		for j in range(DR.shape[1]):
			if DR[i][j] != 0:
				for k in range(DD.shape[0]):#j
					if DD[j][k] != 0 and j != k:
						list.append([1,i,j,k,0])
	'''
	#r-d-r-d
	for i in range(DR.shape[0]):
		for j in range(DR.shape[1]):
			if DR[i][j] != 0:
				for k in range(DR.shape[0]):
					if DR[k][j] != 0 and i != k:
						for m in range(DR.shape[1]):
							if DR[k][m] != 0 and j != m:
								list.append([2,i,j,k,m])

	#r-r-r-d
	for i in range(RR.shape[0]):
		for j in range(RR.shape[1]):
			if RR[i][j] != 0 and i != j:
				for k in range(RR.shape[0]):
					if RR[k][j] != 0 and k != j and k != i:
						for m in range(DR.shape[1]):
							if DR[k][m] != 0:
								list.append([3,i,j,k,m])
	#r-r-d-d
	for i in range(RR.shape[0]):
		for j in range(RR.shape[1]):
			if RR[i][j] != 0 and i != j:
				for k in range(DR.shape[1]):
					if DR[j][k] != 0:
						for m in range(DD.shape[1]):
							if DD[k][m] != 0 and k != m:
								list.append([4,i,j,k,m])
	#r-d-d-d
	for i in range(DR.shape[0]):
		for j in range(DR.shape[1]):
			if DR[i][j] != 0:
				for k in range(DD.shape[1]):
					if DD[j][k] != 0 and j != k:
						for m in range(DD.shape[0]):
							if DD[m][k] != 0 and m != k and m != j:
								list.append([5,i,j,k,m])
	'''
	return list
def ErrAnalysis(A,B):
	#误差分析，两矩阵对应位置相减取绝对值
	err = 0.0
	for i in range(A.shape[0]):
		for j in range(A.shape[1]):
			err += abs(A[i,j] - B[i,j])
	return err
def Normalize(DR):
	'''
	:param DR:待归一化矩阵
	:return: 归一化矩阵
	'''
	line_list = np.sum(DR, axis=1)
	for i in range(DR.shape[0]):
		for j in range(DR.shape[1]):
			DR[i][j] = float(DR[i][j]) / line_list[i]
	return DR
def RandomWalk(MR,MD,DR,alpha):
	'''
	:param MR:593*593
	:param MD: 313*313
	:param DR: 313*593
	:param alpha: 0.1
	:return: 游走之后的
	'''
	Nor_DR = DR = DR / sum(DR)
	MD = Normalize(MD)
	MR = Normalize(MR)
	for i in range(100):
		mid = Nor_DR
		RW_R = np.dot(MD,Nor_DR)*alpha + (1-alpha)*DR
		RW_D = np.dot(Nor_DR,MR)*alpha + (1-alpha)*DR
		Nor_DR = (RW_R + RW_D) / 2
		err = ErrAnalysis(Nor_DR,mid)
		if err < 10**-6:
			break
	return Nor_DR
def splitArray(MatAFilePath):
	Data_list = []
	with open(MatAFilePath, "r") as f:
		for fLine in f:
			row = [int(x) for x in fLine.split()]  # 读出文件，split默认按空格进行分割
			if (len(row) > 0):
				Data_list.append(row)
	Data_array = array(Data_list)
	[irows, icols] = Data_array.shape
	WorkData_list = []
	mid_list = []
	for i in range(irows):
		for j in range(icols):
			if (int(Data_array[i][j]) == 1):
				mid_list.append((i, j))
	random.shuffle(mid_list)  # 打乱排序

	WorkData_list = [mid_list[i:i + (len(mid_list) // 5 + 1)] for i in
					 range(0, len(mid_list), len(mid_list) // 5 + 1)]
	return WorkData_list

def TwoRandomWalk(MR,MD,RD,alpha):
	'''
	:param MR:	药物相似性矩阵,np.array
	:param MD:  疾病相似性矩阵，np.array
	:param RD:  药物疾病关联矩阵，np.array
	:param alpha:  重启概率 0.8
	:return: MR、MD、RD
	'''
	#M1 = np.concatenate((MR,RD),axis = 1)
	M1 = np.hstack((MR,RD))
	M2 = np.hstack((RD.T,MD))
	M = np.vstack((M1,M2)) #拼接后的矩阵
	# 对M进行行归一化
	sum_list = np.sum(M, axis=1)
	for i in range(M.shape[0]):
		for j in range(M.shape[1]):
			M[i][j] = M[i][j] / sum_list[i]
	#初始概率矩阵P，对角为1的矩阵
	P = np.eye(M.shape[0],dtype = int)
	P0 = P/1
	for i in range(100):
		RD = P
		P = (1-alpha)*M.T.dot(P) + alpha*P0
		err = 	la.norm(P-RD)
		if err < 10e-9:
			print("随机游走进行了%d退出"%i)
			break
	# MR = P[0:593,0:593] # 行号   列号
	# RD = P[593:,0:593]
	# MD = P[593:,593:]
	# return MR,MD,RD
	return P

def ChangeArray(A,list,id):
	'''
	:param A:原始药物疾病矩阵
	:param list: 所有1的坐标的list
	:param id: 第id作为test，其他四分作为测试
	:return: 测试矩阵和训练矩阵
	'''
	train_array = 1*A
	test_array = -1*A
	# array = np.array(list)
	for i in range(len(list[id])):
		train_array [ list[id][i][0] ][ list[id][i][1] ] = 0
		test_array [ list[id][i][0] ][ list[id][i][1] ] = 1
	return train_array,test_array


def softmax1(x):
	"""
	Compute the softmax function for each row of the input x.
	Arguments:
	x -- A N dimensional vector or M x N dimensional numpy matrix.
	Return:
	x -- You are allowed to modify x in-place
	"""
	orig_shape = x.shape
	if len(x.shape) > 1:
		# Matrix
		exp_minmax = lambda x: np.exp(x - np.max(x))
		denom = lambda x: 1.0 / np.sum(x)
		x = np.apply_along_axis(exp_minmax, 1, x)
		denominator = np.apply_along_axis(denom, 1, x)
		if len(denominator.shape) == 1:
			denominator = denominator.reshape((denominator.shape[0], 1))
		x = x * denominator
	else:
		# Vector
		x_max = np.max(x)
		x = x - x_max
		numerator = np.exp(x)
		denominator = 1.0 / np.sum(numerator)
		x = numerator.dot(denominator)
	assert x.shape == orig_shape
	# return x
def softmax2(x):
	return np.exp(x) / np.sum(np.exp(x), axis=0)

def NodeAttention():
	pass