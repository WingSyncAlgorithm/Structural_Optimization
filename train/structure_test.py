import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_structure(boundary_matrix, points_matrix, linking_probability):
    """
    生成結構矩陣。

    參數：
    boundary_matrix: 三維矩陣，代表外殼，有材料的部分是1，其他空的地方是0。
    points_matrix: 三維矩陣，代表內部的一些點，有點的位置是1，無點的位置是0。
    linking_probability: 介於0～100的正實數，代表任兩個內部點鍵結的機率。

    返回：
    三維矩陣，結構矩陣，元素只有0或1。
    """
    size = boundary_matrix.shape  # 結構的大小

    # 初始化結構，所有元素設為0或1
    structure = np.zeros(size)

    # 將邊界部分標記為1
    structure[boundary_matrix == 1] = 1

    # 將內部點標記為1，同時確保內部點不在邊界外
    structure[points_matrix == 1] = 1

    # 建立內部點之間的隨機鏈結
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                if points_matrix[i, j, k] == 1 and boundary_matrix[i, j, k] != 1:
                    for x2 in range(size[0]):
                        for y2 in range(size[1]):
                            for z2 in range(size[2]):
                                if points_matrix[x2, y2, z2] == 1 and boundary_matrix[x2, y2, z2] != 1:
                                    # 隨機生成鍵結機率
                                    link_probability = np.random.randint(0, 101)
                                    if link_probability > linking_probability:
                                        # 將兩點之間的元素都設置為1
                                        structure[i, j, k] = 1
                                        structure[x2, y2, z2] = 1

    # 轉換為整數類型
    structure = structure.astype(int)

    return structure

def plot_structure_3d(structure):
    """
    繪製三維結構圖。

    參數：
    structure: 三維矩陣，結構矩陣，元素只有0或1。
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 獲取結構中元素為1的索引
    indices = np.where(structure == 1)

    # 繪製線條
    ax.scatter(indices[0], indices[1], indices[2], c='black', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

    # 打印矩陣
    print("Generated Structure Matrix:")
    print(structure)

# 測試程式
structure_size = 10  # 調整大小
boundary_matrix = np.ones((structure_size, structure_size, structure_size))  # 假設邊界是一個 size x size x size 的正方體
boundary_matrix[1:structure_size-1, 1:structure_size-1, 1:structure_size-1] = 0  # 在內部標記為0

NUM_POINTS = 25
points_matrix = np.zeros((structure_size, structure_size, structure_size), dtype=int)
# 確保隨機點不在邊界上
for _ in range(NUM_POINTS):
    while True:
        x, y, z = np.random.randint(1, structure_size-1, 3)
        if points_matrix[x, y, z] == 0:  # 確保該點尚未被選中
            points_matrix[x, y, z] = 1
            break

linking_probability = 50  # 鍵結的機率為50%

structure = create_structure(boundary_matrix, points_matrix, linking_probability)
plot_structure_3d(structure)
