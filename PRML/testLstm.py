#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-

import torch
import torch.autograd as autograd		#Torch中自动计算梯度模块
import torch.nn as nn					#神经网络模块
import torch.nn.functional as F			#神经网路模块中的常用功能
import torch.optim as optim				#模型优化器模块

torch.manual_seed(1)
lstm = nn.LSTM(3,3)			#LSTM单元输入和输出维度都是3

#生成一个长度为5，每一个元素为1*3的序列作为输入，这里数字3对应于句中的第一个3
inputs = [autograd.Variable(torch.randn((1,3))) for  _ in range(5)]
print('inputs',inputs)	#inputs [tensor([[-0.5525,  0.6355, -0.3968]]), tensor([[-0.6571, -1.6428,  0.9803]]), tensor([[-0.0421, -0.8206,  0.3133]]), tensor([[-1.1352,  0.3773, -0.2824]]), tensor([[-2.5667, -1.4303,  0.5009]])]

#设置隐藏层维度，初始化隐藏层的数据
hidden = (autograd.Variable(torch.randn(1,1,3)) , autograd.Variable(torch.randn(1,1,3)))
print('hidden',hidden)
for i in inputs:
	out,hidden = lstm(i.view(1,1,-1),hidden)
	print(out)
	print(hidden)