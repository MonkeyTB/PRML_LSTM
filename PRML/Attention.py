#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

# import  ipdb
from numpy import *

class LSTM(nn.Module):
	def __init__(self):
		super(LSTM,self).__init__()
		self.lstm = nn.GRU(
			input_size=1444,
			hidden_size=100,
			num_layers=1,
			batch_first=True,
			bidirectional=True,
		)
		# self.out1 = nn.Linear(100*2*4 ,2)
		self.out1 = nn.Linear(100 * 2, 2)
	def forward(self,x):

		r_out,h_n = self.lstm(x,None)
		# out = self.out1(r_out.view(1,4*200))
		out = self.out1(r_out[:,-1,:])
		return out,h_n

class Attention(nn.Module):			#结点attention
	def __init__(self,inputsize=100,outputsize=10):
		super(Attention,self).__init__()
		self.inputsize = inputsize
		self.outputsize = outputsize
		self.node = nn.Linear(self.inputsize,self.outputsize,bias=False)	#过线性层
		init.xavier_uniform(self.node.weight)
		self.u = nn.Linear(self.outputsize,1,bias=False)			#过h(n)的线性层
		init.xavier_uniform(self.u.weight)
	def forward(self,h_hidden):
		# ipdb.set_trace()
		temp_node = self.node(h_hidden)	#W(hs)*h(ij)   4*10
		temp_node = F.tanh(temp_node)	#tanh[W(hs)*h(ij)]
		s_ij = self.u(temp_node)		#h(n)tanh[W(hs)*h(ij)]
		alpha = F.softmax(s_ij)			#4*1
		return torch.mm(alpha.view(1,4),h_hidden)
class Path_Attention(nn.Module):
	def __init__(self,inputsize = 100,outputsize = 10):
		super(Path_Attention,self).__init__()
		self.inputsize = inputsize
		self.outputsize = outputsize
		self.Path = nn.Linear(self.inputsize,self.outputsize,bias = False)
		init.xavier_uniform(self.Path.weight)
		self.u = nn.Linear(self.outputsize,1,bias=False)
		init.xavier_uniform(self.u.weight)
	def forward(self,hidden,num):
		# ipdb.set_trace()
		temp_path = self.Path(hidden)		#num * 100
		temp_path = F.tanh(temp_path)		#tanh(num*100)
		g_ij = self.u(temp_path)			#h(p)tanh[W(ys)*yi]
		return torch.mm(g_ij.view(1, num), hidden)

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
			nn.MaxPool2d(kernel_size=(2,1))
		)
		self.conv2=nn.Sequential(
			nn.Conv2d(16,32,(3,20),1,1),
			nn.ReLU(),
			nn.MaxPool2d((2,1))
		)
		self.out = nn.Linear(32,2)
	def forward(self,x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0),-1)
		output = self.out(x)
		return output
