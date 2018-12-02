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
import ipdb
torch.manual_seed(1)

#--------------------------CNN模型开始-------------------------------------
class CNN(nn.Module):
	def __init__(self):
		super(CNN,self).__init__()
		self.conv1=nn.Sequential(
			nn.Conv2d(
				in_channels=1,
				out_channels=16,
				kernel_size=(3,20),
				stride=1,
				padding=1,
			),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(1,2))
		)
		self.conv2=nn.Sequential(
			nn.Conv2d(16,32,(3,20),1,1),
			nn.ReLU(),
			nn.MaxPool2d((1,2))
		)
		self.out1 = nn.Linear(22272,2000)
		self.out2 = nn.Linear(2000,2)
	def forward(self,x):

		x = self.conv1(x.unsqueeze(0))		#torch.Size([1, 16, 2, 713])
		x = self.conv2(x)					#torch.Size([1, 32, 2, 348])
		x = x.view(x.size(0),-1)
		output = self.out1(x)
		output = self.out2(output)
		return output
cnn = CNN()
if torch.cuda.is_available():
	cnn.cuda()
optimizer = torch.optim.Adam(cnn.parameters(),lr = 0.001)
loss_func = nn.CrossEntropyLoss()
#--------------------------CNN模型结束-------------------------------------
EPOCH = 10
if __name__ == "__main__":
	DrugSim_list = RFTL.readFileToList("..\data\drugSim.txt", 0)
	DisSim_list = RFTL.readFileToList("..\data\disSim.txt", 0)
	DiDr_list = RFTL.readFileToList('..\data\DiDrAMat.txt',1)
	DiDrSplit_list = RFTL.splitArray("..\data\DiDrAMat.txt")	#所有1的坐标的list


	DiDr_array, DiDr_testArray = RFTL.ChangeArray(np.array(DiDr_list), DiDrSplit_list, 0)  # 第一份1做test
	np.savetxt('..\data\第一份训练数据_cnn.txt',DiDr_array, fmt=['%s'] * DiDr_array.shape[1], newline='\n')
	np.savetxt('..\data\第一份测试数据_cnn.txt',DiDr_testArray, fmt=['%s'] * DiDr_testArray.shape[1], newline='\n')
	DiDr = RFTL.TwoRandomWalk(np.array(DrugSim_list),np.array(DisSim_list),DiDr_array,0.9)
	np.savetxt('..\data\随机游走.txt',DiDr, fmt=['%s'] * DiDr.shape[1], newline='\n')
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
	random.seed(100)
	random.shuffle(train_list)
	train_num = 0

	for j in range(EPOCH):
		for i in range(len(train_list)):
			CnnEmbedding_list = EV.CnnEmbeddingVector(DiDr,train_list[i][0],train_list[i][1])
			y_train = []
			y_train.append(DiDr_array[ train_list[i][0] ][ train_list[i][1] ])
			y_train = torch.Tensor(np.array(y_train))
			b_y = torch.Tensor.long(Variable(y_train))
			cnn_x = torch.Tensor(torch.Tensor(np.array(CnnEmbedding_list)))
			if torch.cuda.is_available():
				cnn_x = cnn_x.cuda()
				b_y = b_y.cuda()
			cnn_out = cnn(cnn_x.unsqueeze(0))
			ipdb.set_trace()
			cnn_loss = loss_func(cnn_out,b_y)
			optimizer.zero_grad()
			cnn_loss.backward()
			optimizer.step()
			print('CNN','Epoch:', j, '|train loss:%.4f' % cnn_loss.data[0])
	torch.save(cnn,'..\data\cnn_Module.pkl')	#保存模型

	'''
	#------------------test CNN----------------------------------------	
	'''

	model_cnn = torch.load('..\data\cnn_Module.pkl')  # 加载模型
	DiDr_testCnnArray = []
	for i in range(DiDr_testArray.shape[0]):
		mid = [0 if x == 1 else x for x in DiDr_testArray[i].tolist()]
		DiDr_testCnnArray.append(mid)
	DiDr_testCnnArray = np.array(DiDr_testCnnArray)  # 测试中1 变为-1 后的结果
	prediction_CNN = np.zeros(DiDr_testCnnArray.shape, dtype=float)  # 存储test后的预测结果
	for m in range(DiDr_testCnnArray.shape[0]):
		for n in range(DiDr_testCnnArray.shape[1]):
			CnnEmbedding_list = EV.CnnEmbeddingVector(DiDr, m, n)
			p_cnn = []
			cnn_x = torch.Tensor(torch.Tensor(np.array(CnnEmbedding_list)))
			if torch.cuda.is_available():
				cnn_x = cnn_x.cuda()
			predict_cnn = model_cnn(cnn_x.unsqueeze(0))
			p_cnn = F.softmax(predict_cnn)  # 将i到j的所有路径经过LSTM的测试结果(概率)保存在p中
			prediction_CNN[m][n] = p_cnn[0, 1]
			print('药物%d' % m, '疾病%d' % n, p_cnn[0, 1])
	np.savetxt('..\data\prediction_cnn.txt',prediction_CNN, fmt=['%s'] * prediction_CNN.shape[1], newline='\n')