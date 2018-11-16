#_Author_:Monkey

#!/usr/bin/env python

#-*- coding:utf-8 -*-



import numpy as np

from numpy import *

from __init__ import *

import function as RFTL

import embeddingVector as EV

import Attention as Atten

import math

import ipdb



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

EPOCH = 1
class Prml(nn.Module):
	def __init__(self):
		super(Prml,self).__init__()
		self.BiLSTM = Atten.LSTM()
		self.Attention_Node = Atten.Attention()
		self.Attention_Path = Atten.Path_Attention()
	def forward(self, train_loader):
		for step, (x) in enumerate(train_loader):
			b_x = torch.Tensor.float(Variable(x.view(-1, 4, 1444)))
			if torch.cuda.is_available():
				b_x = b_x.cuda()
			out,(h_n,h_c) = self.BiLSTM(b_x)
			h = torch.stack( (torch.stack ( (torch.mm(out, torch.stack((h_n[0][0],h_n[1][0]),dim = 0)).view(-1,100),
							   torch.mm(out, torch.stack((h_n[0][1], h_n[1][1]), dim=0)).view(-1, 100)),dim = 0) ,	#默认为0,h(1*100)
							torch.stack((torch.mm(out, torch.stack((h_n[0][2], h_n[1][2]), dim=0)).view(-1, 100),
						 		torch.mm(out, torch.stack((h_n[0][3], h_n[1][3]), dim=0)).view(-1, 100)), dim=0) ),dim = 0)
			if step == 0:
				y_i = self.Attention_Node(h.view(4,100))
			else:
				y_i = torch.cat( (y_i,self.Attention_Node(h.view(4,100))),0)
		L = self.Attention_Path(y_i,step+1)			#(step+1)*100  step+1为路径条数
		return L
prml = Prml()
if torch.cuda.is_available():

	prml.cuda()

optimizer = torch.optim.Adam(prml.parameters(),lr = 0.001)

loss_func = nn.CrossEntropyLoss()

# ---------搭建LSTM模型结束----------------



if __name__ == "__main__":

	DrugSim_list = RFTL.readFileToList("..\data\drugSim.txt", 0)

	DisSim_list = RFTL.readFileToList("..\data\disSim.txt", 0)

	DiDr_list = RFTL.readFileToList('..\data\DiDrAMat.txt',1)

	DiDrSplit_list = RFTL.splitArray("..\data\DiDrAMat.txt")	#所有1的坐标的list



	DiDr_array, DiDr_testArray = RFTL.ChangeArray(np.array(DiDr_list), DiDrSplit_list, 0)  # 第一份1做test

	#存路径部分，到时候在放开，目前先执行一次保存文件

	# Pst_list = RFTL.FindStepPath(np.array(DrugSim_list),np.array(DisSim_list),DiDr_array)

	# print(shape(np.array(Pst_list)))

	# np.savetxt('..\data\pst.txt',

	# 		   np.array(Pst_list), fmt=['%s'] * np.array(Pst_list).shape[1], newline='\n')



	DiDr = RFTL.TwoRandomWalk(np.array(DrugSim_list),np.array(DisSim_list),DiDr_array,0.9)

	np.savetxt('..\data\随机游走.txt',

			   DiDr, fmt=['%s'] * DiDr.shape[1], newline='\n')



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

			Node_Node_path_num = len(x_train)

			for epoch in range(EPOCH):
				b_y = torch.Tensor.long(Variable(y_train))
				if torch.cuda.is_available():
					b_y = b_y.cuda()
				output = prml(train_loader)
				loss = loss_func(output, b_y)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				print('Epoch:', epoch, '|train loss:%.4f' % loss.data[0])



	torch.save(prml,'..\data\lstm_Module.pkl')	#保存模型





	model = torch.load('..\data\lstm_Module.pkl')  #加载模型

	DiDr_testPathArray = []

	for i in range(DiDr_testArray.shape[0]):

		mid = [0 if x == 1 else x for x in DiDr_testArray[i].tolist()]

		DiDr_testPathArray.append(mid)

	DiDr_testPathArray = np.array(DiDr_testPathArray)		#测试中1 变为-1 后的结果

	prediction = np.zeros(DiDr_testPathArray.shape,dtype = float) #存储test后的预测结果

	for m in range(DiDr_testPathArray.shape[0]):

		for n in range(DiDr_testPathArray.shape[1]):

			PstFilePath = "..\data\PSTF\(" + str(m) + ")\\" + str(n) + ".txt"

			NoteEmbedding_list = EV.embeddingNoteVector(PstFilePath, DiDr, m, n)

			if len(NoteEmbedding_list) != 0:

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

					# p.append( RFTL.softmax2(predict[0].tolist()).tolist()) #将i到j的所有路径经过LSTM的测试结果(概率)保存在p中

					p.append(F.softmax(predict[0]).tolist())  # 将i到j的所有路径经过LSTM的测试结果(概率)保存在p中

				p = np.sum(np.array(p),axis = 0)	#p = [[0.46,0.53]]

				prediction[m][n] = p[1]

				print('药物%d'%m,'疾病%d'%n,p[1])

		np.savetxt('..\data\prediction.txt',

				   prediction, fmt=['%s'] * prediction.shape[1], newline='\n')