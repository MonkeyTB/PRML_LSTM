#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-

import  numpy as np
from numpy import *
import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
torch.manual_seed(1)



x_train = []
y_train = []

for j in range(len(list)):
	for i in range(len(list[0])-1):
		x_train.append(list[0][i+1])
y_train.append(1)
print(shape(x_train),type(x_train))
train_loader = torch.utils.data.DataLoader(dataset=np.array(x_train),batch_size=4,shuffle=True)
y_train = torch.Tensor(np.array(y_train))
print(torch.typename(train_loader))


EPOCH = 1
for epoch in range(EPOCH):
	for step,(x) in enumerate(train_loader):
		b_x = torch.Tensor.float(Variable(x.view(-1,4,906)))
		b_y = torch.Tensor.long(Variable(y_train))
		if torch.cuda.is_available():
			b_x = b_x.cuda()
			b_y = b_y.cuda()
		output =lstm(b_x)
		loss = loss_func(output,b_y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print('Epoch:', epoch, '|train loss:%.4f' % loss.data[0])

