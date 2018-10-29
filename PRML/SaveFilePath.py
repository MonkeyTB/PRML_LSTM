import numpy as np



Wrr = np.loadtxt("..\data\drugSim.txt")
Wdd = np.loadtxt("..\data\disSim.txt")
Wrd = np.loadtxt("..\data\DiDrAMat.txt")
path = []
drugNum = Wrr.shape[0]
diseaseNum = Wdd.shape[0]
# 路径集输出方式：追加！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

# 0.R-D
print("开始分析路径类型0(一跳)")
# for Rs in range(drugNum):   # 从Rs出发
#     print("第0次路径分析第" + str(Rs) + "个药物")
#     for Dt in range(diseaseNum):
#         if Wrd[Rs, Dt] == 1:    # 走到Dt
#             file = open("..\data\PSTF\(" + str(Rs) + ")\\" + str(Dt) + ".txt", "w+")
#             file.write('{} {} {} {} {}\n'.format(0, Rs, Dt, 0, 0))
#             file.close()

# 1.R-R-D
print("开始分析路径类型1(二跳)")

# for Rs in range(drugNum):   # r-r-d
#     print("第1次路径分析第" + str(Rs) + "个药物")
#     for R1 in range(drugNum):
#         if Rs != R1 and Wrr[Rs, R1] != 0:    # 走到R1(跳过Rs)
#             for Dt in range(diseaseNum):
#                 if Wrd[R1, Dt] == 1:    # 走到Dt
#                     file = open("..\data\PSTF\(" + str(Rs) + ")\\" + str(Dt) + ".txt", "w+")
#                     file.write('{} {} {} {} {}\n'.format(1, Rs, R1, Dt, 0))
#                     file.close()
# 2.R-D-D
for Rs in range(drugNum):   # r-d-d
    print("第2次路径分析第" + str(Rs) + "个药物")
    for D1 in range(diseaseNum):
        if Wrd[Rs, D1] != 0:        # 走到D1
            for Dt in range(diseaseNum):
                if Dt != D1 and Wdd[D1, Dt] != 0:   # 走到Dt(跳过D1）
                    file = open("..\data\PSTF\(" + str(Rs) + ")\\" + str(Dt) + ".txt", "a+")
                    file.write('{} {} {} {} {}\n'.format(2, Rs, D1, Dt, 0))
                    file.close()