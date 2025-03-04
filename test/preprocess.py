import os  # 添加导入os模块
from plyfile import PlyData, PlyElement
import torch
import numpy as np
import random
import trimesh
from torch_scatter import scatter_mean, scatter_max

source_path = "data/shapenet/02871439"
out_path = source_path


for filename in os.listdir(source_path):
    if filename.endswith('.ply'):
        ply_file_path = os.path.join(source_path, filename)
        plydata = PlyData.read(ply_file_path)
        if len(plydata.elements[0]["x"]) > 40000:
            continue
        coord = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1, dtype=np.float32)

        features_dc = np.stack((np.asarray(plydata.elements[0]["f_dc_0"]),
                                np.asarray(plydata.elements[0]["f_dc_1"]),
                                np.asarray(plydata.elements[0]["f_dc_2"])), axis=1, dtype=np.float32)
        
        opacities = np.asarray(plydata.elements[0]["opacity"], dtype=np.float32)
        
        scales = np.stack((np.asarray(plydata.elements[0]["scale_0"]),
                           np.asarray(plydata.elements[0]["scale_1"]),
                           np.asarray(plydata.elements[0]["scale_2"])), axis=1, dtype=np.float32)

        rots = np.stack((np.asarray(plydata.elements[0]["rot_0"]),
                         np.asarray(plydata.elements[0]["rot_1"]),
                         np.asarray(plydata.elements[0]["rot_2"]),
                         np.asarray(plydata.elements[0]["rot_3"])), axis=1, dtype=np.float32)
        
        # if not 'min_max_dict' in globals():
        #     global min_max_dict
        #     min_max_dict = {
        #         'coord': {'min': None, 'max': None},
        #         'features_dc': {'min': None, 'max': None},
        #         'opacities': {'min': None, 'max': None},
        #         'scales': {'min': None, 'max': None},
        #         'rots': {'min': None, 'max': None}
        #     }
        
        # # 更新各特征统计值
        # def update_stats(feature_name, data):
        #     current_min = np.min(data, axis=0)
        #     current_max = np.max(data, axis=0)
            
        #     if min_max_dict[feature_name]['min'] is None:
        #         min_max_dict[feature_name]['min'] = current_min
        #         min_max_dict[feature_name]['max'] = current_max
        #     else:
        #         min_max_dict[feature_name]['min'] = np.minimum(
        #             min_max_dict[feature_name]['min'], current_min)
        #         min_max_dict[feature_name]['max'] = np.maximum(
        #             min_max_dict[feature_name]['max'], current_max)
        
        # update_stats('coord', coord)
        # update_stats('features_dc', features_dc)
        # update_stats('opacities', opacities)
        # update_stats('scales', scales)
        # update_stats('rots', rots)

        # continue
        
        features = np.concatenate((coord, features_dc, opacities[:, None], scales, rots), axis=1, dtype=np.float32)
        
        coord_tensor = torch.from_numpy(coord)
        features_tensor = torch.from_numpy(features)

        # 四舍五入处理坐标（使用PyTorch实现）
        rounded_coord = torch.round(coord_tensor * 1000) / 1000  # 保留3位小数
        
        # 创建唯一坐标的哈希键
        unique_coord, inverse_indices = torch.unique(
            rounded_coord, 
            dim=0, 
            return_inverse=True, 
            return_counts=False
        )

        # 坐标取平均值
        avg_coord = scatter_mean(coord_tensor, inverse_indices, dim=0)
        
        # 特征取最大值（保留每个通道的最大值）
        max_features, _ = scatter_max(features_tensor, inverse_indices, dim=0)
        
        # 转换回numpy数组
        coord = avg_coord.numpy()
        features = max_features.numpy()
        
        # 拆分特征到各个字段
        x = coord[:,0]
        y = coord[:,1]
        z = coord[:,2]
        f_dc = features[:, 3:6]
        opacity = features[:, 6]
        scale = features[:, 7:10]
        rot = features[:, 10:14]

        # 填充数据前添加精度处理
        f_dc = np.round(f_dc, 3)    
        opacity = np.round(opacity, 2)  
        scale = np.round(scale, 2)      
        rot = np.round(rot, 3)          


        # 创建新的PlyElement
        vertex_elements = np.zeros(len(x), dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
            ('opacity', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
        ])
        
        # 填充数据
        vertex_elements['x'] = x
        vertex_elements['y'] = y
        vertex_elements['z'] = z
        vertex_elements['f_dc_0'], vertex_elements['f_dc_1'], vertex_elements['f_dc_2'] = f_dc.T
        vertex_elements['opacity'] = opacity
        vertex_elements['scale_0'], vertex_elements['scale_1'], vertex_elements['scale_2'] = scale.T
        vertex_elements['rot_0'], vertex_elements['rot_1'], vertex_elements['rot_2'], vertex_elements['rot_3'] = rot.T

        # 创建PlyData并保存
        ply_data = PlyData([PlyElement.describe(vertex_elements, 'vertex')])
        output_path = os.path.join(out_path, f"Pre-{filename}")
        ply_data.write(output_path)

# print(min_max_dict)