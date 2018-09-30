#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-

from __init__ import *
import numpy as np
from numpy import *
import function as RFTL


def embeddingNoteVector(FilePath,DrugSim_array,DisSim_array,DiDrCorr_array,r,d):
	'''
	:param FilePath: 节点路径文件的路径
	:param DrugSim_array: 阈值后的药物相似矩阵
	:param DisSim_array: 阈值后的疾病相似矩阵
	:return: 
	'''
	embeddingVector_list = []
	Pst_list = RFTL.readFileToList(FilePath, 1)
	Pst_array = np.array(Pst_list)
	flag = False
	for i in range(Pst_array.shape[0]):
		# print( Pst_array[i][1],type(DrugSim_array[ Pst_array[i][1] ] ) )
		mid_list = []
		if Pst_array[i][0] == 0 or Pst_array[i][0] == 1:
			if Pst_array[i][1] == r and  Pst_array[i][3] == d:
				mid_list.append([Pst_array[i][1], Pst_array[i][3]])  # 起始药物ID和终止疾病ID
				flag = True
		else :
			if Pst_array[i][1] == r and Pst_array[i][4] == d:
				mid_list.append([Pst_array[i][1], Pst_array[i][4]])  # 起始药物ID和终止疾病ID
				flag = True
		if flag == True:
			mid_list.append(DrugSim_array[Pst_array[i][1]].tolist() + DiDrCorr_array.T[Pst_array[i][1]].tolist())  # 第一种药物
			if  Pst_array[i][0] == 0:		# r-r-d
				mid_list.append(DrugSim_array[ Pst_array[i][2] ].tolist() + DiDrCorr_array.T[ Pst_array[i][2] ].tolist())	#第二种药物
				mid_list.append(DiDrCorr_array[ Pst_array[i][3] ].tolist() + DisSim_array[ Pst_array[i][3] ].tolist())	#第一种疾病
				mid_list.append([0]*(DrugSim_array.shape[0] + DisSim_array.shape[0]))					#最后元素补0
				embeddingVector_list.append(mid_list)
			if  Pst_array[i][0] == 1:		# r-d-d
				mid_list.append(DiDrCorr_array[ Pst_array[i][2] ].tolist() + DisSim_array[ Pst_array[i][2] ].tolist())	#第一种疾病
				mid_list.append(DiDrCorr_array[ Pst_array[i][3] ].tolist() + DisSim_array[ Pst_array[i][3] ].tolist())	#第二种疾病
				mid_list.append([0] * (DrugSim_array.shape[0] + DisSim_array.shape[0]))  # 最后元素补0
				embeddingVector_list.append(mid_list)
			if  Pst_array[i][0] ==  2:		# r-d-r-d
				mid_list.append(DiDrCorr_array[Pst_array[i][2]].tolist() + DisSim_array[Pst_array[i][2]].tolist())  	# 第一种疾病
				mid_list.append(DrugSim_array[Pst_array[i][3]].tolist() + DiDrCorr_array.T[Pst_array[i][3]].tolist()) 	# 第二种药物
				mid_list.append(DiDrCorr_array[Pst_array[i][4]].tolist() + DisSim_array[Pst_array[i][4]].tolist())  	# 第二种疾病
				embeddingVector_list.append(mid_list)
			if  Pst_array[i][0] == 3:		# r-r-r-d
				mid_list.append(DrugSim_array[Pst_array[i][2]].tolist() + DiDrCorr_array.T[Pst_array[i][2]].tolist())  # 第二种药物
				mid_list.append(DrugSim_array[Pst_array[i][3]].tolist() + DiDrCorr_array.T[Pst_array[i][3]].tolist())  # 第三种药物
				mid_list.append(DiDrCorr_array[Pst_array[i][4]].tolist() + DisSim_array[Pst_array[i][4]].tolist())	   # 第一种疾病
				embeddingVector_list.append(mid_list)
			if  Pst_array[i][0] == 4:		# r-r-d-d
				mid_list.append(DrugSim_array[Pst_array[i][2]].tolist() + DiDrCorr_array.T[Pst_array[i][2]].tolist())  # 第二种药物
				mid_list.append(DiDrCorr_array[Pst_array[i][3]].tolist() + DisSim_array[Pst_array[i][3]].tolist())     # 第一种疾病
				mid_list.append(DiDrCorr_array[Pst_array[i][4]].tolist() + DisSim_array[Pst_array[i][4]].tolist())     # 第一种疾病
				embeddingVector_list.append(mid_list)
			if  Pst_array[i][0] == 5:		# r-d-d-d
				mid_list.append(DiDrCorr_array[Pst_array[i][2]].tolist() + DisSim_array[Pst_array[i][2]].tolist())     # 第一种疾病
				mid_list.append(DiDrCorr_array[Pst_array[i][3]].tolist() + DisSim_array[Pst_array[i][3]].tolist())     # 第二种疾病
				mid_list.append(DiDrCorr_array[Pst_array[i][4]].tolist() + DisSim_array[Pst_array[i][4]].tolist())     # 第三种疾病
				embeddingVector_list.append(mid_list)
		flag = False
	return embeddingVector_list


