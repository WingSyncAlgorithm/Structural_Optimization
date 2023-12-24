import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class StructureVisualizer:
    def create_structure(self, boundary, points, probability, show_percentage=0.1):
        """
        創建三維結構。

        參數:
        - boundary (numpy.ndarray): 代表外殼的三維矩陣。
        - points (numpy.ndarray): 代表內部點的三維矩陣。
        - probability (float): 內部點連接的機率。
        - show_percentage (float): 在可視化中顯示的內部點的百分比。

        返回:
        - 結構 (numpy.ndarray): 代表生成結構的三維矩陣。
        """

        def connect_points(structure, point1, point2, connected_points):
            """
            連接結構中的兩個點。

            參數:
            - structure (numpy.ndarray): 代表結構的三維矩陣。
            - point1 (tuple): 第一個點的坐標。
            - point2 (tuple): 第二個點的坐標。
            - connected_points (list): 存儲連接點的列表。

            返回:
            - structure (numpy.ndarray): 連接點後的更新結構。
            - connected_points (list): 更新後的連接點列表。
            """
            structure[point1] = 1
            structure[point2] = 1
            connected_points.append((point1, point2))
            return structure, connected_points

        def boundary_point(index, shape):
            """
            將一維索引轉換為三維坐標。

            參數:
            - index (int): 一維索引。
            - shape (tuple): 三維矩陣的形狀。

            返回:
            - tuple: 代表三維坐標的元組。
            """
            dim_x, dim_y, dim_z = shape
            z = index // (dim_x * dim_y)
            y = (index - z * dim_x * dim_y) // dim_x
            x = index - z * dim_x * dim_y - y * dim_x
            return x, y, z

        def visualize_structure(structure, connected_points):
            """
            將結構可視化。

            參數:
            - structure (numpy.ndarray): 代表結構的三維矩陣。
            - connected_points (list): 存儲連接點的列表。
            """
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

            x, y, z = np.where(structure == 1)
            num_points = len(x)
            num_points_to_show = int(show_percentage * num_points)
            indices_to_show = np.random.choice(num_points, num_points_to_show, replace=False)

            ax.scatter(x[indices_to_show], y[indices_to_show], z[indices_to_show], color='black', s=100)

            for connection in connected_points:
                point1, point2 = connection
                ax.scatter([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], color='gray', s=100)

            plt.show()

        # 初始化結構矩陣
        structure = np.zeros_like(boundary)
        structure += boundary

        # 儲存連接的點以進行可視化
        connected_points = []

        # 根據機率連接內部點
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if np.random.rand() * 100 < probability:
                    structure, connected_points = connect_points(structure, points[i], points[j], connected_points)

        # 以較低的機率連接邊界點和內部點
        for _ in range(10):
            i = np.random.randint(0, len(boundary))
            j = np.random.randint(0, len(points))
            if np.random.rand() > 0.0001:
                structure, connected_points = connect_points(
                    structure, boundary_point(i, boundary.shape), points[j], connected_points
                )

        # 將結構矩陣和連接的點進行可視化
        visualize_structure(structure, connected_points)

        return structure

# 主要使用
if __name__ == "__main__":
    # 定義結構的維度
    SIZE = 10

    # 創建邊界矩陣（立方體）
    BOUNDARY = np.zeros((SIZE, SIZE, SIZE))
    BOUNDARY[0:SIZE, 0:SIZE, 0:SIZE] = 1

    # 創建隨機的內部點
    POINTS = np.random.randint(1, SIZE - 1, size=(10, 3))

    # 設定內部點之間連接的機率
    INTERNAL_PROBABILITY = 50

    # 創建結構
    visualizer = StructureVisualizer()
    visualizer.create_structure(BOUNDARY, POINTS, INTERNAL_PROBABILITY)
