#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
from numpy import *
from __init__ import *
import function as RFTL
import embeddingVector as EV
import math

import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import datetime
import torch.nn.functional as F
torch.manual_seed(1)

startTime,endTime = 0,0
print(torch.__version__)
# -------------搭建LSTM模型-------------
EPOCH = 3
class LSTM(nn.Module):
	def __init__(self):
		super(LSTM,self).__init__()
		self.lstm = nn.LSTM(
			input_size=1444,
			hidden_size=400,
			num_layers=1,
			batch_first=True,
			bidirectional=True,
		)
		self.out1 = nn.Linear(400*2,100)
		self.out2 = nn.Linear(100,2)


	def forward(self,x):
		# print(torch.typename(x))
		r_out,(h_n,h_c) = self.lstm(x,None)
		out = self.out1(r_out[:,-1,:])
		out = self.out2(out)


		return out

lstm = LSTM()
if torch.cuda.is_available():
	lstm.cuda()
optimizer = torch.optim.Adam(lstm.parameters(),lr = 0.001)
loss_func = nn.CrossEntropyLoss()
# ---------搭建LSTM模型结束----------------

if __name__ == "__main__":
	DrugSim_list = RFTL.readFileToList("..\data\drugSim.txt", 0)
	DisSim_list = RFTL.readFileToList("..\data\disSim.txt", 0)
	DiDr_list = RFTL.readFileToList('..\data\DiDrAMat.txt',1)
	DiDrSplit_list = RFTL.splitArray("..\data\DiDrAMat.txt")	#所有1的坐标的list
	# DrugSim_array = RFTL.changeArray(np.array(DrugSim_list),VPT)
	# DisSim_array = RFTL.changeArray(np.array(DisSim_list),VPT)
	DiDr_array, DiDr_testArray = RFTL.ChangeArray(np.array(DiDr_list), DiDrSplit_list, 0)  # 第一份1做test
	#存路径部分，到时候在放开，目前先执行一次保存文件
	# Pst_list = RFTL.FindStepPath(np.array(DrugSim_list),np.array(DisSim_list),DiDr_array)
	# print(shape(np.array(Pst_list)))
	# np.savetxt('..\data\pst.txt',
	# 		   np.array(Pst_list), fmt=['%s'] * np.array(Pst_list).shape[1], newline='\n')

	# # 随机游走
	#DiDrCorr_array = RFTL.RandomWalk(DrugSim_array,DisSim_array,DiDr_array,0.1)

	DiDr = RFTL.TwoRandomWalk(np.array(DrugSim_list),np.array(DisSim_list),DiDr_array,0.9)
	np.savetxt('..\data\随机游走.txt',
			   DiDr, fmt=['%s'] * DiDr.shape[1], newline='\n')

	# PstFilePath = '..\data\pst.txt'


	#训练的坐标，包括四分1 和 随机四分0
	train_list = DiDrSplit_list[1] + DiDrSplit_list[2] + DiDrSplit_list[3] + DiDrSplit_list[4]

	total,num = len(train_list),0
	while True:
		x,y = random.randint(0,np.array(DiDr_list).shape[0]),random.randint(0,np.array(DiDr_list).shape[1])
		if DiDr_list[x][y] == 0:
			train_list.append((x,y))
			num += 1
		if num == total:
			break

	random.shuffle(train_list)
	# print('train_list',train_list)
	train_num = 0

	for i in range(len(train_list)):
		print('第 %d 组数据训练'%train_num,'第%d次进入训练'%i,'训练标签%d'%DiDr_array[ train_list[i][0] ][ train_list[i][1] ])
		startTime = datetime.datetime.now()
		PstFilePath = "..\data\PSTF\(" + str(train_list[i][0]) + ")\\" + str(train_list[i][1]) + ".txt"
		NoteEmbedding_list = EV.embeddingNoteVector( PstFilePath,DiDr ,train_list[i][0],train_list[i][1])
		endTime = datetime.datetime.now()
		print('寻找一组节点嵌入向量耗时%d' % (endTime - startTime).seconds)
		if len(NoteEmbedding_list) != 0:
			train_num += 1
			print('药物%d'%train_list[i][0],'疾病%d'%train_list[i][1],len(NoteEmbedding_list))
			x_train = []
			y_train = []

			for m in range(len(NoteEmbedding_list)):
				for n in range(len(NoteEmbedding_list[0]) - 1):
					x_train.append(NoteEmbedding_list[0][n + 1])
			y_train.append(DiDr_array[ train_list[i][0] ][ train_list[i][1] ])
			train_loader = torch.utils.data.DataLoader(dataset=np.array(x_train), batch_size=4, shuffle=True)
			y_train = torch.Tensor(np.array(y_train))

			for epoch in range(EPOCH):
				for step, (x) in enumerate(train_loader):
					b_x = torch.Tensor.float(Variable(x.view(-1, 4, 1444)))
					b_y = torch.Tensor.long(Variable(y_train))
					if torch.cuda.is_available():
						b_x = b_x.cuda()
						b_y = b_y.cuda()
					output = lstm(b_x)
					loss = loss_func(output, b_y)
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
					print('Epoch:', epoch, '|train loss:%.4f' % loss.data[0])

	torch.save(lstm,'..\data\lstm_ModuleD4_1.pkl')	#保存模型



	model = torch.load('..\data\lstm_ModuleD4_1.pkl')  #加载模型
	DiDr_testPathArray = []
	for i in range(DiDr_testArray.shape[0]):
		mid = [0 if x == 1 else x for x in DiDr_testArray[i].tolist()]
		DiDr_testPathArray.append(mid)
	DiDr_testPathArray = np.array(DiDr_testPathArray)
	prediction = np.zeros(DiDr_testPathArray.shape,dtype = float) #存储test后的预测结果
	for m in range(DiDr_testPathArray.shape[0]):
		for n in range(DiDr_testPathArray.shape[1]):
			PstFilePath = "..\data\PSTF\(" + str(m) + ")\\" + str(n) + ".txt"
			NoteEmbedding_list = EV.embeddingNoteVector(PstFilePath, DiDr, m, n)
			if len(NoteEmbedding_list) != 0:
				print('药物%d'%m,'疾病%d'%n)
				x_train = []
				for i in range(len(NoteEmbedding_list)):
					for j in range(len(NoteEmbedding_list[0]) - 1):
						x_train.append(NoteEmbedding_list[0][j + 1])
				train_loader = torch.utils.data.DataLoader(dataset=np.array(x_train), batch_size=4, shuffle=True)
				p = []
				for step, (x) in enumerate(train_loader):
					b_x = torch.Tensor.float(Variable(x.view(-1, 4, 1444)))
					if torch.cuda.is_available():
						b_x = b_x.cuda()
					predict = model(b_x)
					p.append( RFTL.softmax2(predict[0].tolist()).tolist()) #将i到j的所有路径经过LSTM的测试结果(概率)保存在p中
				p = np.mean(np.array(p),axis = 0)	#p = [[0.46,0.53]]
				prediction[m][n] = p[1]
		np.savetxt('..\data\predictionD4_1.txt',
				   prediction, fmt=['%s'] * prediction.shape[1], newline='\n')
