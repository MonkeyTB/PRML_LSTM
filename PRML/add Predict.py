#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import function as FUN

FileC_Path = '..\data\Batch50E10LR0.001.pkl.npy'
FileG_Path = '..\data\GRUM.txt'

cnn_list = np.load(FileC_Path)
gru_list = FUN.readFileToList(FileG_Path,0)

predict = 0.6*cnn_list + 0.4*np.array(gru_list)

np.savetxt('..\data\prediction.txt',

		   predict, fmt=['%s'] * predict.shape[1], newline='\n')
