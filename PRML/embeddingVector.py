#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-

from __init__ import *
import numpy as np
from numpy import *
import function as RFTL

# import ipdb


def embeddingNoteVector(FilePath,DiDr,r,d):
	'''
	:param FilePath: 节点路径文件的路径
	:param DiDr:游走完后的拼接矩阵
	:return:
	'''
	# embeddingVector_list = []
	# Pst_array = np.array( RFTL.readFileToList(FilePath, 1) )
	# flag = False
	# for i in range(Pst_array.shape[0]):
	# 	# print( Pst_array[i][1],type(DrugSim_array[ Pst_array[i][1] ] ) )
	# 	mid_list = []
	# 	if Pst_array[i][0] == 0 or Pst_array[i][0] == 1:
	# 		if Pst_array[i][1] == r and  Pst_array[i][3] == d:
	# 			mid_list.append([Pst_array[i][1], Pst_array[i][3]])  # 起始药物ID和终止疾病ID
	# 			flag = True
	# 	# else :
	# 	# 	if Pst_array[i][1] == r and Pst_array[i][4] == d:
	# 	# 		mid_list.append([Pst_array[i][1], Pst_array[i][4]])  # 起始药物ID和终止疾病ID
	# 	# 		flag = True
	# 	if flag == True:
	# 		mid_list.append(DiDr[ Pst_array[i][1] ])  # 第一种药物
	# 		if  Pst_array[i][0] == 0:		# r-r-d
	# 			mid_list.append( DiDr[ Pst_array[i][2] ])	#第二种药物
	# 			mid_list.append( DiDr[ Pst_array[i][3] + 593 ])	#第一种疾病
	# 			mid_list.append( [0]*DiDr.shape[0] )					#最后元素补0
	# 			embeddingVector_list.append(mid_list)
	# 		if  Pst_array[i][0] == 1:		# r-d-d
	# 			mid_list.append(DiDr[ Pst_array[i][2] + 593 ])	#第一种疾病
	# 			mid_list.append(DiDr[ Pst_array[i][3] + 593 ])	#第二种疾病
	# 			mid_list.append([0] * DiDr.shape[0] )  # 最后元素补0
	# 			embeddingVector_list.append(mid_list)
			# if  Pst_array[i][0] ==  2:		# r-d-r-d
			# 	mid_list.append(DiDr[ Pst_array[i][2] + 593 ] )  	# 第一种疾病
			# 	mid_list.append(DiDr[ Pst_array[i][3] ]) 	# 第二种药物
			# 	mid_list.append(DiDr[ Pst_array[i][4] + 593 ])  	# 第二种疾病
			# 	embeddingVector_list.append(mid_list)
			# if  Pst_array[i][0] == 3:		# r-r-r-d
			# 	mid_list.append(DiDr[Pst_array[i][2]])  # 第二种药物
			# 	mid_list.append(DiDr[Pst_array[i][3]])  # 第三种药物
			# 	mid_list.append(DiDr[Pst_array[i][4] + 593 ])	   # 第一种疾病
			# 	embeddingVector_list.append(mid_list)
			# if  Pst_array[i][0] == 4:		# r-r-d-d
			# 	mid_list.append(DiDr[Pst_array[i][2]])  # 第二种药物
			# 	mid_list.append(DiDr[Pst_array[i][3] + 593 ])     # 第一种疾病
			# 	mid_list.append(DiDr[Pst_array[i][4] + 593])     # 第一种疾病
			# 	embeddingVector_list.append(mid_list)
			# if  Pst_array[i][0] == 5:		# r-d-d-d
			# 	mid_list.append(DiDr[Pst_array[i][2] + 593 ])     # 第一种疾病
			# 	mid_list.append(DiDr[Pst_array[i][3] + 593 ])     # 第二种疾病
			# 	mid_list.append(DiDr[Pst_array[i][4] + 593 ])     # 第三种疾病
			# 	embeddingVector_list.append(mid_list)？
	# 	flag = False
	# return embeddingVector_list
	embeddingVector_list = []
	Pst_array = np.array( RFTL.readFileToList(FilePath, 1) )
	flag = False


	if Pst_array.shape[0] != 0:		#文件不为空
		for i in range(Pst_array.shape[0]):
			mid_list = []
			if Pst_array[i][1] == r and  Pst_array[i][3] == d:
				mid_list.append([Pst_array[i][1], Pst_array[i][3]])  # 起始药物ID和终止疾病ID
				flag = True
			if flag == True:
				mid_list.append(DiDr[ Pst_array[i][1] ])  # 第一种药物
				if Pst_array[i][0] == 0:
					mid_list.append( DiDr[ Pst_array[i][2] ])
					mid_list.append( [0]*DiDr.shape[0])
					mid_list.append( [0]*DiDr.shape[0] )					#最后元素补0
					embeddingVector_list.append(mid_list)
				if  Pst_array[i][0] == 1:		# r-r-d
					mid_list.append( DiDr[ Pst_array[i][2] ])	#第二种药物
					mid_list.append( DiDr[ Pst_array[i][3] + 763 ])	#第一种疾病
					mid_list.append( [0]*DiDr.shape[0] )					#最后元素补0
					embeddingVector_list.append(mid_list)
				if  Pst_array[i][0] == 2:		# r-d-d
					mid_list.append(DiDr[ Pst_array[i][2] + 763 ])	#第一种疾病
					mid_list.append(DiDr[ Pst_array[i][3] + 763 ])	#第二种疾病
					mid_list.append([0] * DiDr.shape[0] )  # 最后元素补0
					embeddingVector_list.append(mid_list)

			flag = False
	'''
		mid_list = []
		mid_list.append([r, d])  # 起始药物ID和终止疾病ID
		mid_list.append(DiDr[r])  # 第一种药物
		mid_list.append(DiDr[d + 763])  # 第一种疾病
		mid_list.append([0] * DiDr.shape[0])  # 最后元素补0
		mid_list.append([0] * DiDr.shape[0])  # 最后元素补0
		embeddingVector_list.append(mid_list)
	else:		#文件为空
		mid_list = []
		mid_list.append([r, d])  # 起始药物ID和终止疾病ID
		mid_list.append(DiDr[r])  # 第一种药物
		mid_list.append(DiDr[d + 763])  # 第一种疾病
		mid_list.append([0] * DiDr.shape[0])  # 最后元素补0
		mid_list.append([0] * DiDr.shape[0])  # 最后元素补0
		embeddingVector_list.append(mid_list)
	'''
	return embeddingVector_list