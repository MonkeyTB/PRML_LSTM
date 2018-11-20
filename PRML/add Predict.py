#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import function as FUN

FileC_Path = '..\data\prediction_CNN.txt'
FileG_Path = '..\data\prediction_lstm.txt'

cnn_list = FUN.readFileToList(FileC_Path,0)
gru_list = FUN.readFileToList(FileG_Path,0)

predict = 0.9*np.array(cnn_list) + 0.1*np.array(gru_list)

np.savetxt('..\data\prediction.txt',

		   predict, fmt=['%s'] * predict.shape[1], newline='\n')
