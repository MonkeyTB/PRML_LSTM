#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-
import  numpy as np
import operator
A = [
	[1,2,3,4],
	[5,6,7,8],
	[9,10,11,12],
	[13,14,15,16],
	[17,18,19,20],
]
B = [
	[1,2,3,4],
	[9,10,11,12],
	[17,18,19,20],
]
A = np.array(A)
B = np.array(B)
del_list = []
for i in range(A.shape[0]):
	for j in range(B.shape[0]):
		if set(A[i]) == set(B[j]):
			del_list.append(i)
A = np.delete(A, del_list,0)  # 删除A的第i行

print('end ',A)
