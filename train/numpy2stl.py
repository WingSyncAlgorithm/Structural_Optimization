import numpy as np
from stl import mesh

def numpy_array_to_stl(numpy_array, stl_path):
    '''
    Arags:
        numpy_array: 3維numpy陣列
        stl_path: 要創建的stl檔案路徑
    Returns:
        None
    '''
    
    mesh_data = mesh.Mesh(np.zeros(numpy_array.shape[0], dtype=mesh.Mesh.dtype))
    
    # ['v0', 'v1', 'v2'] 分別是三個頂點
    for i in range(numpy_array.shape[0]):
        vertices = numpy_array[['v0', 'v1', 'v2']][i]  
        for j in range(3):
            mesh_data.vectors[i][j] = vertices[j]  
    

    mesh_data.save(stl_path)

def generate_cube_array(side_length=1.0):
    '''
    生成立方體numpy array 的函數
    這邊只用來測試
    Args:
        side_length : 立方體邊長
    Return : 立方體的numpy array
    '''
    # 定義立方體的八個頂點座標
    vertices = np.array([
        [0, 0, 0],  # V0
        [side_length, 0, 0],  # V1
        [side_length, side_length, 0],  # V2
        [0, side_length, 0],  # V3
        [0, 0, side_length],  # V4
        [side_length, 0, side_length],  # V5
        [side_length, side_length, side_length],  # V6
        [0, side_length, side_length]  # V7
    ])

    # 將頂點座標組成三角面片，這裡以立方體的每個面都分成兩個三角形
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # 底面
        [4, 5, 6], [4, 6, 7],  # 頂面
        [0, 4, 7], [0, 7, 3],  # 側面1
        [1, 5, 6], [1, 6, 2],  # 側面2
        [0, 1, 5], [0, 5, 4],  # 側面3
        [2, 6, 7], [2, 7, 3]   # 側面4
    ])

    # 組合成最終的 NumPy 數組表示立方體
    cube_array = np.zeros(faces.shape[0], dtype=[('normals', np.float32, (3,)),
                                                  ('v0', np.float32, (3,)),
                                                  ('v1', np.float32, (3,)),
                                                  ('v2', np.float32, (3,))])

    for i in range(faces.shape[0]):
        v0, v1, v2 = vertices[faces[i, :]]
        normal = np.cross(v1 - v0, v2 - v0)
        normal /= np.linalg.norm(normal)
        cube_array['normals'][i] = normal
        cube_array['v0'][i] = v0
        cube_array['v1'][i] = v1
        cube_array['v2'][i] = v2

    return cube_array

cube_array = generate_cube_array(side_length=1.0)

numpy_array_to_stl(cube_array, r"C:\Users\s9909\Desktop\stl\output.stl")
