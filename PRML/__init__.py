#_Author_:Monkey
#!/usr/bin/env python
#-*- coding:utf-8 -*-




DrugSim_list = []	#药物相似性
DisSim_list = []	#疾病相似性
DiDrCorr_list = []	#药物疾病相关性
VPT = 0.2			#阈值
Pst_list = []		#节点路径
NoteEmbedding_list = [] # 节点嵌入向量
''''
Pst_list保存路径,第一个元素是
		0代表r-r-d,
		1代表r-d-d,
		2代表r-d-r-d,
		3代表r-r-r-d,
		4代表r-r-d-d,
		5代表r-d-d-d
'''