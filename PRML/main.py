#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
from numpy import *
from __init__ import *
import function as RFTL
import embeddingVector as EV


import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
torch.manual_seed(1)

print(torch.__version__)
# -------------搭建LSTM模型-------------
EPOCH = 1
class LSTM(nn.Module):
	def __init__(self):
		super(LSTM,self).__init__()
		self.lstm = nn.LSTM(
			input_size=906,
			hidden_size=4,
			num_layers=1,
			batch_first=True,
			bidirectional=True,
		)
		self.out = nn.Linear(8,2)
	def forward(self,x):
		print(torch.typename(x))
		r_out,(h_n,h_c) = self.lstm(x,None)
		out = self.out(r_out[:,-1,:])
		return out

lstm = LSTM()
if torch.cuda.is_available():
	lstm.cuda()
optimizer = torch.optim.Adam(lstm.parameters(),lr = 0.001)
loss_func = nn.CrossEntropyLoss()
# ---------搭建LSTM模型结束----------------

if __name__ == "__main__":
	DrugSim_list = RFTL.readFileToList("..\data\DrugSimMat", 0)
	DisSim_list = RFTL.readFileToList("..\data\DiseaseSimMat", 0)

	DiDrCorr_dict = RFTL.splitArray("..\data\DiDrAMat")

	DrugSim_array = RFTL.changeArray(np.array(DrugSim_list),VPT)
	DisSim_array = RFTL.changeArray(np.array(DisSim_list),VPT)

	for name in sorted(DiDrCorr_dict):
		if name == 'train1_array':
			DiDr_array = DiDrCorr_dict[name]
		if name == 'test1_array':
			DiDe_testArray = DiDrCorr_dict[name]

	#存路径部分，到时候在放开，目前先执行一次保存文件
	# Pst_list = RFTL.FindStepPath(DrugSim_array,DisSim_array,DiDr_array)
	# print(shape(np.array(Pst_list)))
	# np.savetxt('..\data\pst.txt',
	# 		   np.array(Pst_list), fmt=['%s'] * np.array(Pst_list).shape[1], newline='\n')


	# # 随机游走
	#DiDrCorr_array = RFTL.RandomWalk(DrugSim_array,DisSim_array,DiDr_array,0.1)
	DrugSim_array,DisSim_array,DiDrCorr_array = RFTL.TwoRandomWalk(DrugSim_array,DisSim_array,DiDr_array,0.8)
	PstFilePath = '..\data\pst.txt'
	for i in range(DrugSim_array.shape[0]):
		for j in range(DisSim_array.shape[0]):
			NoteEmbedding_list = EV.embeddingNoteVector( PstFilePath, DrugSim_array, DisSim_array, DiDrCorr_array ,i,j)
			if len(NoteEmbedding_list) != 0:
				print(len(NoteEmbedding_list))
				x_train = []
				y_train = []

				for m in range(len(NoteEmbedding_list)):
					for n in range(len(NoteEmbedding_list[0]) - 1):
						x_train.append(NoteEmbedding_list[0][n + 1])
				y_train.append(DiDr_array[i][j])
				train_loader = torch.utils.data.DataLoader(dataset=np.array(x_train), batch_size=4, shuffle=True)
				y_train = torch.Tensor(np.array(y_train))

				for epoch in range(EPOCH):
					for step, (x) in enumerate(train_loader):
						b_x = torch.Tensor.float(Variable(x.view(-1, 4, 906)))
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



	torch.save(lstm,'..\data\lstm_Module.pkl')	#保存模型
	# model = torch.load('..\data\lstm_Module.pkl')  #加载模型
	# np.savetxt('..\data\\NoteEmbedding.txt',
	# 		   np.array(NoteEmbedding_list), fmt=['%s'] * np.array(NoteEmbedding_list).shape[1], newline='\n')

