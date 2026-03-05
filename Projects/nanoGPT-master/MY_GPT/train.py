import numpy as np
import matplotlib.pyplot as plt

# 1. 定义参数 r 和 theta 的范围
# linspace(start, end, number_of_points)
r = np.linspace(0, 5, 50)
theta = np.linspace(0, 2 * np.pi, 80)

# 2. 从参数创建网格
# 这会在 r-theta 平面上创建一个网格
R, THETA = np.meshgrid(r, theta)

# 3. 应用参数方程计算 x, y, z 坐标
X = R * np.cos(THETA)
Y = R * np.sin(THETA)
Z = R # z = r

# 4. 绘制 3D 曲面
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap='viridis', # 使用 viridis 颜色映射
                       linewidth=0, antialiased=True)

ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')
ax.set_title('Parametric Cone: z = r')

# 设置坐标轴比例一致，看起来更像一个圆锥
ax.set_aspect('equal')

plt.show()