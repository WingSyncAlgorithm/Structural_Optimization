import numpy as np

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

    # 遍歷內部點，同時確保內部點不在邊界外
    internal_points_indices = np.argwhere(points_matrix == 1)
    structure[internal_points_indices[:, 0], internal_points_indices[:, 1], internal_points_indices[:, 2]] = 1

    # 建立內部點之間的隨機鏈結
    for i in range(len(internal_points_indices)):
        for j in range(i+1, len(internal_points_indices)):
            point1, point2 = internal_points_indices[i], internal_points_indices[j]

            # 隨機生成鍵結機率
            link_probability = np.random.randint(0, 101)
            if link_probability > linking_probability:
                # 使用 Bresenham3D 來連接內部點
                line_structure = bresenham_3d(*point1, *point2)
                structure[line_structure[:, 0], line_structure[:, 1], line_structure[:, 2]] = 1

    # 轉換為整數類型
    structure = structure.astype(int)

    return structure

def bresenham_3d(x1, y1, z1, x2, y2, z2):
    ListOfPoints = []
    ListOfPoints.append((x1, y1, z1))
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    
    if (x2 > x1):
        xs = 1
    else:
        xs = -1
    if (y2 > y1):
        ys = 1
    else:
        ys = -1
    if (z2 > z1):
        zs = 1
    else:
        zs = -1

    # Initialize a list to store all points on the line
    line_points = []

    # Driving axis is X-axis"
    if (dx >= dy and dx >= dz):     
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while (x1 != x2):
            x1 += xs
            if (p1 >= 0):
                y1 += ys
                p1 -= 2 * dx
            if (p2 >= 0):
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            line_points.append((x1, y1, z1))

    # Driving axis is Y-axis"
    elif (dy >= dx and dy >= dz):     
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while (y1 != y2):
            y1 += ys
            if (p1 >= 0):
                x1 += xs
                p1 -= 2 * dy
            if (p2 >= 0):
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            line_points.append((x1, y1, z1))

    # Driving axis is Z-axis"
    else:     
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while (z1 != z2):
            z1 += zs
            if (p1 >= 0):
                y1 += ys
                p1 -= 2 * dz
            if (p2 >= 0):
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            line_points.append((x1, y1, z1))

    # Convert the list of points to a numpy array
    line_points = np.array(line_points)

    return line_points

# 測試程式
structure_size = 10  # 調整大小
boundary_matrix = np.ones((structure_size, structure_size, structure_size))
boundary_matrix[1:structure_size-1, 1:structure_size-1, 1:structure_size-1] = 0

NUM_POINTS = 25
points_matrix = np.zeros((structure_size, structure_size, structure_size), dtype=int)

for _ in range(NUM_POINTS):
    while True:
        x, y, z = np.random.randint(1, structure_size-1, 3)
        if points_matrix[x, y, z] == 0:
            points_matrix[x, y, z] = 1
            break

linking_probability = 50

structure = create_structure(boundary_matrix, points_matrix, linking_probability)
print("Generated Structure Matrix:")
print(structure)
