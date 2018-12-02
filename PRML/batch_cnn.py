import numpy as np

import torch

from torch.utils.data import *

import torch.nn as nn  # 神经网络模块

import torch.utils.data as Data

import batch_cnn_predict as CNNP



BATCH_SIZE = 50

EPOCH = 10

LR = 0.001

groupID = 0

MODELNAME = "Batch" + str(BATCH_SIZE) + "E" + str(EPOCH) + "LR" + str(LR)
import torch.nn as nn  # 神经网络模块

import torch





# ======================================================================CNN模型搭建

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=(3, 20),
                      stride=1,
                      padding=(1,10)
                      ),
            nn.ReLU(),
            nn.MaxPool2d((1,2)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,(3,20),1,(1,10)),
            nn.ReLU(),
            nn.MaxPool2d((1,2)),
        )
        self.out = nn.Linear(23104, 2)
    def forward(self, x):
        x = self.conv1(x)  # [batch, 1, 2, 1444] -> [batch, 16, 2, 722]
        x = self.conv2(x)  # [batch, 16, 2, 722] -> [batch, 32, 2, 361]
        # x = torch.Tensor.tanh(x)
        b, n_f, h, w = x.shape
        output = self.out(x.view(b, n_f * h * w))   #[50*23104] -> [50*2]
        return output

model = CNN()  # 实例化神经网络模型(双向LSTM)
if torch.cuda.is_available():  # 尝试GPU加速
    model = model.cuda()
loss_func = nn.CrossEntropyLoss()  # 损失函数使用CrossEntropyLoss
optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # 优化器选用


def getFeature(xySet,Wrr,Wdd,Wrd,drugNum,diseaseNum):
    A1 = np.hstack((Wrr, Wrd))
    A2 = np.hstack((Wrd.T, Wdd))
    A = np.vstack((A1, A2))
    train_x = []
    train_y = []
    for item in xySet:
        train_x.append(np.vstack((A[item[0]], A[item[1] + drugNum])))
        train_y.append(Wrd[item[0], item[1]])
    return np.array(train_x), np.array(train_y)

# =================================================================train

def train(train_array):
    Wrr = np.loadtxt("..\data\drugSim.txt")
    Wdd = np.loadtxt("..\data\disSim.txt")
    Wrd = np.loadtxt("..\data\第一份训练数据.txt")
    drugNum = Wrr.shape[0]  # 药物数量
    diseaseNum = Wdd.shape[0]  # 疾病数量

    train_x, train_y = getFeature(train_array,Wrr,Wdd,Wrd,drugNum,diseaseNum)

    dataset = Data.TensorDataset(torch.from_numpy(train_x),torch.from_numpy(train_y))
    dataLoader = Data.DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True)

    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(dataLoader):
            batch_x = torch.Tensor.float(batch_x)
            if torch.cuda.is_available():
                # 将训练数据放入GPU
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()

            output = model(batch_x.view(-1, 1, 2, drugNum + diseaseNum))
            loss = loss_func(output, torch.Tensor.long(batch_y))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                print("     第{}趟第{}组训练完成，loss = {}".format(epoch, step, loss.data[0]))

    torch.save(model, "..\\data\\" + MODELNAME + ".pkl")  # 保存训练模型

    CNNP.test("..\\data\\" + MODELNAME + ".pkl", BATCH_SIZE)







