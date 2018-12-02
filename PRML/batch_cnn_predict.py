#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-


import numpy as np

import torch

import torch.nn as nn  # 神经网络模块

import torch.nn.functional as F  # 神经网络模块中的常用功能

import time



import torch.utils.data as Data

start = time.clock()        # 开始计时

def getFeature(dr,Wrr,Wdd,Wrd,drugNum,diseaseNum):
    A1 = np.hstack((Wrr, Wrd))
    A2 = np.hstack((Wrd.T, Wdd))
    A = np.vstack((A1, A2))
    test_x = []
    test_y = []
    for i in range(diseaseNum):
        test_x.append((A[dr], A[i + drugNum]))
    return np.array(test_x)
def test(modelname, batch_size):
    Wrr = np.loadtxt("..\data\drugSim.txt")
    Wdd = np.loadtxt("..\data\disSim.txt")
    Wrd = np.loadtxt("..\data\第一份测试数据.txt")

    drugNum = Wrr.shape[0]  # 药物数量
    diseaseNum = Wdd.shape[0]  # 疾病数量

    model = torch.load("..\\data\\" + modelname )  # 加载训练好的神经网络模型
    if torch.cuda.is_available():       # 尝试GPU加速
        model = model.cuda()
    print(model)
    result = np.zeros(Wrd.shape)
    for dr in range(drugNum):
        di_flag = 0     # 标志当前操作的疾病坐标
        test_x = getFeature(dr,Wrr,Wdd,Wrd,drugNum,diseaseNum)     # 按药物（行）取测试集
        dataset = Data.TensorDataset(torch.from_numpy(test_x))
        dataLoader = Data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
        )
        for step, batch_x in enumerate(dataLoader):
            batch_x = torch.Tensor.float(batch_x[0])
            if torch.cuda.is_available():
                # 将训练数据放入GPU
                batch_x = batch_x.cuda()
            output = model(batch_x.view(-1, 1, 2, drugNum + diseaseNum))
            result_sorce = nn.functional.softmax(output).data.cpu().numpy().T[1]
            for s in result_sorce:
                result[dr, di_flag] = s
                di_flag += 1
        if dr % 50 == 0:
            print("第{}个药物，当前程序运行了{}分钟".format(dr, (time.clock() - start) / 60))
    np.save("..\\data\\" + modelname + ".npy", result)

