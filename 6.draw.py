import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d

plt.rcParams['font.size'] = 20
plt.figure(figsize=(6,4))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
fm = matplotlib.font_manager.FontProperties()
fm.set_size(22)


# 1. 意愿模型
# fig = plt.figure(figsize=(6, 6))
# x = np.linspace(0, 120, 120)
# y = np.linspace(0, 1, 100)
# xx, yy = np.meshgrid(x, y)
# z = 0.5*( 1 - xx / 120 ) + 0.5*( 0.5*np.cos(yy*np.pi) + 0.5 )
# ax1 = plt.axes(projection='3d')
# surf = ax1.plot_surface(xx, yy, z, cstride=1, rstride=1, cmap=plt.get_cmap('rainbow'), antialiased=False)

# ax1.set_xlabel("Distance")
# ax1.set_ylabel("RCI") # Roadway Congestion Index
# ax1.set_zlabel("Willingness")

# plt.show()

# 2. 车辆数目与覆盖率的关系
# x = [10, 20, 30, 40, 50]
# AAA = [0.1933, 0.3367, 0.41, 0.45, 0.6067]
# BBB = [0.2033, 0.3567, 0.43, 0.4833, 0.6267]
# CCC = [0.23, 0.3733, 0.47, 0.4967, 0.6333]
# TOIM = [0.2467, 0.4033, 0.4833, 0.51, 0.6533]

# plt.plot(x, AAA, label="MWSP", color="b", marker="^")
# plt.plot(x, BBB, label="MTVCS", color="orange", marker="D")
# plt.plot(x, CCC, label="Hector", color="g", marker="x")
# plt.plot(x, TOIM, label="TOIM", color="r", marker="o")
# plt.xlabel("The Number of Workers")
# plt.ylabel("Sensing Coverage")
# # plt.ylim(160, 250)
# plt.xticks(x)
# plt.legend(frameon=False, fontsize='x-small', ncol=1)
# plt.tight_layout()
# plt.show()

# # 3. 车辆数目与单位覆盖成本的关系
# x = [10, 20, 30, 40, 50]
# AAA = [26.4594, 13.117, 9.5083, 6.0627, 4.3124]
# BBB = [14.9006, 7.8336, 5.7144, 3.7965, 3.1892]
# CCC = [9.3285, 3.589, 2.3532, 1.9955, 1.8085]
# TOIM = [4.2934, 3.3001, 1.7413, 1.6831, 1.6268]

# plt.plot(x, AAA, label="MWSP", color="b", marker="^")
# plt.plot(x, BBB, label="MTVCS", color="orange", marker="D")
# plt.plot(x, CCC, label="Hector", color="g", marker="x")
# plt.plot(x, TOIM, label="TOIM", color="r", marker="o")
# plt.xlabel("The Number of Workers")
# plt.ylabel("Unit Covering Cost")
# # plt.ylim(160, 250)
# plt.xticks(x)
# plt.legend(frameon=False, fontsize='x-small', ncol=1)
# plt.tight_layout()
# plt.show()

# 4. 任务数目与覆盖率的关系
# x = [100, 200, 300, 400, 500]
# AAA = [1, 0.755, 0.4867, 0.355, 0.276]
# BBB = [1, 0.805, 0.5, 0.405, 0.328]
# CCC = [1, 0.835, 0.54, 0.4375, 0.358]
# TOIM = [1, 0.835, 0.57, 0.4725, 0.396]

# plt.plot(x, AAA, label="MWSP", color="b", marker="^")
# plt.plot(x, BBB, label="MTVCS", color="orange", marker="D")
# plt.plot(x, CCC, label="Hector", color="g", marker="x")
# plt.plot(x, TOIM, label="TOIM", color="r", marker="o")
# plt.xlabel("The Number of Tasks")
# plt.ylabel("Sensing Coverage")
# # plt.ylim(160, 250)
# plt.xticks(x)
# plt.legend(frameon=False, fontsize='x-small', ncol=1)
# plt.tight_layout()
# plt.show()

# 5. 任务数目与单位覆盖成本的关系
# x = [100, 200, 300, 400, 500]
# AAA = [1.4307, 3.6732, 6.2604, 8.1414, 10.5648]
# BBB = [1.1873, 2.4613, 4.043, 4.9183, 5.6903]
# CCC = [1.0632, 1.7685, 2.0116, 2.3956, 2.5387]
# TOIM = [0.9118, 1.4757, 1.6103, 2.0387, 1.9657]

# plt.plot(x, AAA, label="MWSP", color="b", marker="^")
# plt.plot(x, BBB, label="MTVCS", color="orange", marker="D")
# plt.plot(x, CCC, label="Hector", color="g", marker="x")
# plt.plot(x, TOIM, label="TOIM", color="r", marker="o")
# plt.xlabel("The Number of Tasks")
# plt.ylabel("Unit Covering Cost")
# # plt.ylim(160, 250)
# plt.xticks(x)
# plt.legend(frameon=False, fontsize='x-small', ncol=1)
# plt.tight_layout()
# plt.show()