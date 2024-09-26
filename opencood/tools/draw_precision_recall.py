import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# 假设我们有一些离散的点
x = np.array([0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3])
y = np.array([0.1022, 0.2303, 0.5970, 0.8234, 0.8969, 0.9193, 0.9510, 0.9770, 0.9875, 0.9928])
y2 = np.array([0.4759, 0.2959, 0.2094, 0.1754, 0.1561, 0.1445, 0.1295, 0.1248, 0.1200, 0.1152])
y3 = np.array([0.9447,0.9447,0.9447,0.9696,0.9788,0.9856,0.9932,0.9995,0.9997,1.0000])
y4 = np.array([0.9391,0.9391,0.9391,0.8893,0.8029,0.7004,0.5754,0.3976,0.1729,0.0163])

x2 = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# 使用 CubicSpline 进行样条插值
cs1 = CubicSpline(x, y)
cs2 = CubicSpline(x, y2)
cs3 = CubicSpline(x2, y3)
cs4 = CubicSpline(x2, y4)

# 生成更多的点用于绘制平滑曲线
x_new = np.linspace(min(x), max(x), 100)
x2_new = np.linspace(min(x2), max(x2), 100)
y_new = cs1(x_new)
y2_new = cs2(x_new)
y3_new = cs3(x2_new)
y4_new = cs4(x2_new)

# 绘制原始离散点和插值后的平滑曲线
plt.scatter(x, y, color='red')
plt.plot(x_new, y_new, label='precision_zero-shot')

plt.scatter(x, y2, color='green')
plt.plot(x_new, y2_new, label='recall_zero-shot')

plt.xlabel('score')  # 给 x 轴添加标签
plt.ylabel('')  # 给 y 轴添加标签
plt.legend()
plt.show()

plt.scatter(x2, y3, color='blue')
plt.plot(x2_new, y3_new, label='precision_full')

plt.scatter(x2, y4, color='black')
plt.plot(x2_new, y4_new, label='recall_full')


plt.xlabel('score')  # 给 x 轴添加标签
plt.ylabel('')  # 给 y 轴添加标签
plt.legend()
plt.show()
