import numpy as np
import torch
import networkx as nx


newface_token = 0
stopface_token = 1
padface_token = 2


def get_shifted_sequence(sequence):
    non_special = np.flatnonzero(np.isin(sequence, [0, 1, 2], invert=True))
    if non_special.shape[0] > 0:
        idx = non_special[0]
        val = sequence[idx]
        sequence[non_special] -= (val - 3)
    return sequence


def read_faces(text):
    all_lines = text.splitlines()
    all_face_lines = [x for x in all_lines if x.startswith('f ')]
    all_faces = [[int(y.split('/')[0]) - 1 for y in x.strip().split(' ')[1:]] for x in all_face_lines]
    return all_faces


def read_vertices(text):
    all_lines = text.splitlines()
    all_vertex_lines = [x for x in all_lines if x.startswith('v ')]
    all_vertices = np.array([[float(y) for y in x.strip().split(' ')[1:]] for x in all_vertex_lines])
    assert all_vertices.shape[1] == 3, 'vertices should have 3 coordinates'
    return all_vertices

# 将坐标值限制在特定范围内
def quantize_coordinates(coords, num_tokens=256):
    if torch.is_tensor(coords):
        coords = torch.clip((coords + 0.5), 0, 1) * num_tokens  # type: ignore
        coords_quantized = coords.round().long()
    else:
        coords = np.clip((coords + 0.5), 0, 1) * num_tokens  # type: ignore
        coords_quantized = coords.round().astype(int)
    return coords_quantized

# 找到图中的基本环(不能再分解的环)
def face_to_cycles(face):
    """Find cycles in face."""
    g = nx.Graph()
    for v in range(len(face) - 1):
        g.add_edge(face[v], face[v + 1])
    g.add_edge(face[-1], face[0])
    return list(nx.cycle_basis(g)) # 返回图中所有基本循环

# 量化 去重 排序
def sort_vertices_and_faces(vertices_, faces_, num_tokens=256):
    # 量化（离散化）顶点 使得x上被离散化为128类，y z也是，总计128^3个类别
    # 不改变形状 使得所有值被限制在0 ~ num_tokes之间 这里*操作在numpy数组上 是逐元素相乘
    vertices = np.clip((vertices_ + 0.5), 0, 1) * num_tokens  
    vertices_quantized_ = vertices.round().astype(int) # 四舍五入转换为int
    vertices_quantized_ = vertices_quantized_[:, [2, 1, 0]] # 高级索引重新排列数组 x,y,z -> z,y,x
    
    # 量化后可能出现重复值
    # 去除重复顶点 并返回旧列表元素在新列表的索引 
    vertices_quantized, unique_inverse = np.unique(vertices_quantized_, axis=0, return_inverse=True)
    
    # 按照z,y,x重新排序
    sort_inds = np.lexsort(vertices_quantized.T) # 多级排序顶点 返回了排序完后每个顶点在原数组的索引
    vertices_quantized = vertices_quantized[sort_inds]
    vertices_quantized = np.stack([vertices_quantized[:, 2], vertices_quantized[:, 1], vertices_quantized[:, 0]], axis=-1)

    # 根据排序后的顶点重新索引面 确保面中的顶点索引与排序后的顶点数组一致
    faces = [np.argsort(sort_inds)[unique_inverse[f]] for f in faces_] 
    # 这意味着 np.argsort(sort_inds)[i] 将返回在排序后的顶点数组中，原始索引为 i 的顶点在新排序中的位置
    # i在sort_inds中的下标为np.argsort(sort_inds)，而sort_inds中的下标意味着新排序的顺序，所以np.argsort(sort_inds)是i在新排序中的位置
    

    # Merging duplicate vertices and re-indexing the faces causes some faces to
    # contain loops (e.g [2, 3, 5, 2, 4]). Split these faces into distinct
    # sub-faces.
    # 处理重复面
    sub_faces = []
    for f in faces:
        cliques = face_to_cycles(f)
        for c in cliques:
            c_length = len(c)
            # Only append faces with more than two verts.
            if c_length > 2:
                d = np.argmin(c) # 获得最小值的下标
                # 通过循环索引，使得面中第一个顶点索引是最小的
                sub_faces.append([c[(d + i) % c_length] for i in range(c_length)])
    faces = sub_faces
    # 对每个面进行排序，排序规则为按照最低点索引大小排序，(若相同则按下一个索引)
    faces.sort(key=lambda f: tuple(sorted(f)))

    # 去除face中未引用的顶点
    num_verts = vertices_quantized.shape[0] # 顶点数量
    vert_connected = np.equal( # [:,None]将一维np数组转换为二维列向量
    # 第一个数组为顶点的索引 第二个数组为face中所有索引展开 两个数组形状为(n,1) (1,m)自动广播 形成了邻接矩阵
    # # 沿着列的方向any 即求n个顶点有多少连接到面上了 
        np.arange(num_verts)[:, None], np.hstack(faces)[None]).any(axis=-1) 
    vertices_quantized = vertices_quantized[vert_connected]

    # 重新索引faces以匹配去除未引用点后的顶点数组
    # cumsum计算前缀和，代表左移值 对原始索引左移
    vert_indices = (
            np.arange(num_verts) - np.cumsum(1 - vert_connected.astype('int')))
    faces = [vert_indices[f].tolist() for f in faces]
    
    # 重新归一化顶点
    vertices = vertices_quantized / num_tokens - 0.5
    # order: Z, Y, X --> X, Y, Z
    vertices = np.stack([vertices[:, 2], vertices[:, 1], vertices[:, 0]], axis=-1)
    return vertices, faces