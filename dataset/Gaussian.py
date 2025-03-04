from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import torch
import hydra
from plyfile import PlyData, PlyElement
import random
import trimesh
from collections.abc import Mapping, Sequence
from torch.utils.data.dataloader import default_collate
import os

class GaussianDataset(Dataset):
    def __init__(self,config,split='train',max_sh_degree=3):
        super(GaussianDataset, self).__init__()
        self.split = split
        self.data_root = Path('data',config.dataset,config.category,split)
        self.data_list = self.get_data_list(self.data_root)
        self.discretize = config.ce_output

        self.max_rotation = config.max_rotation  # 最大旋转角度（度）
        self.scale_range = eval(config.scale_range)  # 缩放范围
        self.color_jitter = config.color_jitter  # 颜色扰动强度
        self.density_dropout = config.density_dropout  # 密度丢弃概率
        

        self.bin_config = {
            'coord': {'num_bins': config.coord_bins, 'min_max': (-0.5, 0.5)},      
            'color': {'num_bins': config.color_bins, 'min_max': (-3, 6)},        
            'opacity': {'num_bins': config.opacity_bins, 'min_max': (-6, 12)},        
            'scale': {'num_bins': config.scale_bins, 'min_max': (-17, -2)},        
            'rot': {'num_bins': config.rot_bins, 'min_max': (-2, 4)}           
        }

    def prepare_data(self,idx):
        data_dict = self.get_data(idx)
        data_dict = self.transform_data(data_dict)

        if self.discretize:
            data_dict = self._discretize_gaussian(data_dict)
            if False:
                data_dict = self.reconstruct_from_discrete(data_dict)
                self._save_gaussian_ply(data_dict,idx)
                exit(-1)

        return data_dict

    def __getitem__(self, idx):
        return self.prepare_data(idx)

    def __len__(self):
        return len(self.data_list)

    def get_data_list(self,data_root):
        data_list = list(data_root.glob('*'))  
        return data_list

    def get_data(self, idx):
        path = self.data_list[idx]  
        plydata = PlyData.read(path)
        
        coord = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)

        features_dc = np.stack((np.asarray(plydata.elements[0]["f_dc_0"]),
                                np.asarray(plydata.elements[0]["f_dc_1"]),
                                np.asarray(plydata.elements[0]["f_dc_2"])), axis=1)
        
        opacities = np.asarray(plydata.elements[0]["opacity"])
        
        scales = np.stack((np.asarray(plydata.elements[0]["scale_0"]),
                           np.asarray(plydata.elements[0]["scale_1"]),
                           np.asarray(plydata.elements[0]["scale_2"])), axis=1)
        
        rots = np.stack((np.asarray(plydata.elements[0]["rot_0"]),
                         np.asarray(plydata.elements[0]["rot_1"]),
                         np.asarray(plydata.elements[0]["rot_2"]),
                         np.asarray(plydata.elements[0]["rot_3"])), axis=1)
    
        features = np.concatenate((coord, features_dc, opacities[:, None], scales, rots), axis=1)
        
        data_dict = dict(coord=coord, feat=features)
        
        return data_dict
    
    def transform_data(self, data_dict):
        for key in data_dict:
            data_dict[key] = torch.tensor(data_dict[key])
    

        if self.split == 'train':
            # 保持几何一致性的旋转
            rotation_matrix = self._get_random_rotation()
            data_dict = self._apply_rotation(data_dict, rotation_matrix)
            
            # 保持比例一致的缩放
            scale_factor = self._get_random_scale()
            data_dict = self._apply_scale(data_dict, scale_factor)
            
            # 颜色空间扰动（保持能量守恒）
            data_dict = self._apply_color_jitter(data_dict)
            
            # 高斯球密度扰动（保持总能量不变）
            data_dict = self._apply_density_jitter(data_dict)

            # 控制数据精度
            data_dict = self._apply_accuracy_control(data_dict)
    
        data_dict['offset'] = torch.tensor([len(data_dict['coord'])])

        return data_dict
    
    def _save_gaussian_ply(self,data_dict,idx):
        path = self.data_list[idx]
        print(path)
        out_path = "test"
        # 将处理后的数据转换回ply格式
        processed_feat = data_dict['feat'].numpy()
        
        # 拆分特征到各个字段
        x = processed_feat[:, 0]
        y = processed_feat[:, 1]
        z = processed_feat[:, 2]
        f_dc = processed_feat[:, 3:6]
        opacity = processed_feat[:, 6]
        scale = processed_feat[:, 7:10]
        rot = processed_feat[:, 10:14]


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
        output_path = os.path.join(out_path, f"augmented_test.ply")
        ply_data.write(output_path)

    def _get_random_rotation(self):
        """生成随机的合理旋转矩阵"""
        angle = torch.deg2rad(torch.tensor(random.uniform(-self.max_rotation, self.max_rotation)))
        axis_idx = random.choice([0, 1, 2])
        axis = torch.zeros(3)
        axis[axis_idx] = 1.0
        return torch.tensor(trimesh.transformations.rotation_matrix(angle, axis[:3])[:3, :3],dtype=torch.float32)


    def _apply_rotation(self, data, rot_mat):
        """应用旋转到坐标和旋转参数"""
        # 旋转坐标
        data['coord'] = data['coord'] @ rot_mat.T

        # 调整旋转参数（四元数旋转）
        quats = data['feat'][:, 10:14]  # rot_0~rot_3
        rotated_quats = self._rotate_quaternions(quats, rot_mat)
        data['feat'][:, 10:14] = rotated_quats
        return data

    def _rotate_quaternions(self, quats, rot_mat):
        """根据旋转矩阵调整四元数"""
        # 将旋转矩阵转换为四元数
        rot_quat = torch.tensor(trimesh.transformations.quaternion_from_matrix(rot_mat))
        # 四元数乘法（q_rot * q_original）
        return torch.stack([
            rot_quat[0]*quats[:,0] - rot_quat[1]*quats[:,1] - rot_quat[2]*quats[:,2] - rot_quat[3]*quats[:,3],
            rot_quat[0]*quats[:,1] + rot_quat[1]*quats[:,0] + rot_quat[2]*quats[:,3] - rot_quat[3]*quats[:,2],
            rot_quat[0]*quats[:,2] - rot_quat[1]*quats[:,3] + rot_quat[2]*quats[:,0] + rot_quat[3]*quats[:,1],
            rot_quat[0]*quats[:,3] + rot_quat[1]*quats[:,2] - rot_quat[2]*quats[:,1] + rot_quat[3]*quats[:,0]
        ], dim=1)

    def _get_random_scale(self):
        return random.uniform(*self.scale_range)

    def _apply_scale(self, data, scale):
        """应用缩放并保持物理属性一致"""
        # 缩放坐标
        data['coord'] *= scale

        # 调整缩放参数（保持相对比例）
        scales = data['feat'][:, 7:10]  # scale_0~scale_2
        scales *= scale
        data['feat'][:, 7:10] = scales
        
        # 调整透明度保持能量守恒：opacity *= scale^3 (体积变化)
        data['feat'][:, 6] *= scale**3
        return data

    def _apply_color_jitter(self, data):
        """颜色扰动（在球谐系数空间）"""
        # 只扰动DC项（基础颜色）
        dc_features = data['feat'][:, 3:6]
        jitter = torch.normal(0, self.color_jitter, size=(3,))
        dc_features += jitter
        # 保持能量守恒：归一化
        dc_norm = torch.norm(dc_features, dim=1, keepdim=True)
        dc_features = dc_features / (dc_norm + 1e-6) * (dc_norm + self.color_jitter * torch.rand_like(dc_norm))
        data['feat'][:, 3:6] = dc_features
        return data

    def _apply_density_jitter(self, data):
        """高斯球密度扰动（保持总能量）"""
        num_points = len(data['coord'])
        drop_mask = torch.rand(num_points) > self.density_dropout  # 随机丢弃10%的点
        
        # 调整保留点的透明度补偿能量损失
        data['feat'][~drop_mask, 6] *= 1.0 / (1.0 - self.density_dropout)  # opacity补偿
        
        # 应用mask
        data['coord'] = data['coord'][drop_mask]
        data['feat'] = data['feat'][drop_mask]
        return data

    def _apply_accuracy_control(self,data):
        data['coord'] = torch.round(data['coord'] * 1000) / 1000   
        data['feat'][:, 3:6] = torch.round(data['feat'][:, 3:6] * 1000) / 1000 
        data['feat'][:, 6] = torch.round(data['feat'][:, 6] * 100) / 100 
        data['feat'][:, 7:10] = torch.round(data['feat'][:, 7:10] * 100) / 100 
        data['feat'][:, 10:14] = torch.round(data['feat'][:, 10:14] * 1000) / 1000     
        data['feat'][:,0:3] = data['coord']
        return data

    def _discretize_gaussian(self, data_dict):
        """将高斯特征离散化为类别索引"""
        # 定义各特征的分箱参数（可根据实际数据分布调整）
        bin_config = self.bin_config

        # 获取特征张量
        feat = data_dict['feat']
        
        # 坐标离散化 (前3维)
        coord_bins = torch.linspace(*bin_config['coord']['min_max'], bin_config['coord']['num_bins'])
        coord_indices = torch.bucketize(feat[:, :3], coord_bins) - 1  # bucketize返回索引从1开始
        coord_indices = coord_indices.clamp(0, bin_config['coord']['num_bins']-1)

        # 颜色特征离散化 (3-6维)
        color_bins = torch.linspace(*bin_config['color']['min_max'], bin_config['color']['num_bins'])
        color_indices = torch.bucketize(feat[:, 3:6], color_bins) - 1
        color_indices = color_indices.clamp(0, bin_config['color']['num_bins']-1)
        
        # 透明度离散化 (第6维)
        opacity_bins = torch.linspace(*bin_config['opacity']['min_max'], bin_config['opacity']['num_bins'])
        opacity_indices = torch.bucketize(feat[:, 6], opacity_bins) - 1
        opacity_indices = opacity_indices.clamp(0, bin_config['opacity']['num_bins']-1)
        

        # 缩放系数离散化 (7-9维)
        scale_bins = torch.linspace(*bin_config['scale']['min_max'], bin_config['scale']['num_bins'])
        scale_indices = torch.bucketize(feat[:, 7:10], scale_bins) - 1
        scale_indices = scale_indices.clamp(0, bin_config['scale']['num_bins']-1)


        # 旋转参数离散化 (10-13维)
        rot_bins = torch.linspace(*bin_config['rot']['min_max'], bin_config['rot']['num_bins'])
        rot_indices = torch.bucketize(feat[:, 10:14], rot_bins) - 1
        rot_indices = rot_indices.clamp(0, bin_config['rot']['num_bins']-1)


        # 合并所有离散特征
        discrete_feat = torch.cat([
            coord_indices,
            color_indices,
            opacity_indices.unsqueeze(1),
            scale_indices,
            rot_indices
        ], dim=1).long()

        # 更新数据字典
        data_dict['target'] = discrete_feat

        return data_dict

    def reconstruct_from_discrete(self, data_dict):
        """从离散索引重建原始数据（近似值）"""        
        # 获取离散特征和分箱配置
        discrete_feat = data_dict['target']
        bin_config = self.bin_config

        # 初始化重建后的特征张量
        reconstructed = torch.zeros(discrete_feat.shape[0], 14, dtype=torch.float32)

        # 坐标重建 (前3维)
        coord_bins = torch.linspace(*bin_config['coord']['min_max'], bin_config['coord']['num_bins'] + 1)
        coord_values = (coord_bins[1:] + coord_bins[:-1]) / 2  # 取分箱中点
        reconstructed[:, :3] = coord_values[discrete_feat[:, :3].clamp(0, len(coord_values)-1)]

        # 颜色特征重建 (3-6维)
        color_bins = torch.linspace(*bin_config['color']['min_max'], bin_config['color']['num_bins'] + 1)
        color_values = (color_bins[1:] + color_bins[:-1]) / 2
        reconstructed[:, 3:6] = color_values[discrete_feat[:, 3:6].clamp(0, len(color_values)-1)]

        # 透明度重建 (第6维)
        opacity_bins = torch.linspace(*bin_config['opacity']['min_max'], bin_config['opacity']['num_bins'] + 1)
        opacity_values = (opacity_bins[1:] + opacity_bins[:-1]) / 2
        reconstructed[:, 6] = opacity_values[discrete_feat[:, 6].clamp(0, len(opacity_values)-1)]

        # 缩放系数重建 (7-9维)
        scale_bins = torch.linspace(*bin_config['scale']['min_max'], bin_config['scale']['num_bins'] + 1)
        scale_values = (scale_bins[1:] + scale_bins[:-1]) / 2
        reconstructed[:, 7:10] = scale_values[discrete_feat[:, 7:10].clamp(0, len(scale_values)-1)]

        # 旋转参数重建 (10-13维)
        rot_bins = torch.linspace(*bin_config['rot']['min_max'], bin_config['rot']['num_bins'] + 1)
        rot_values = (rot_bins[1:] + rot_bins[:-1]) / 2
        reconstructed[:, 10:14] = rot_values[discrete_feat[:, 10:14].clamp(0, len(rot_values)-1)]

        # 更新数据字典
        data_dict['feat'] = reconstructed
        return data_dict

def collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
        for key in batch.keys():
            if "offset" in key:
                batch[key] = torch.cumsum(batch[key], dim=0)
        return batch
    else:
        return default_collate(batch)


def point_collate_fn(batch,mix_prob=0):
    assert isinstance(
        batch[0], Mapping
    )  # currently, only support input_dict, rather than input_list
    batch = collate_fn(batch)
    if "offset" in batch.keys():
        # Mix3d (https://arxiv.org/pdf/2110.02210.pdf)
        if random.random() < mix_prob:
            batch["offset"] = torch.cat(
                [batch["offset"][1:-1:2], batch["offset"][-1].unsqueeze(0)], dim=0
            )
    return batch

@hydra.main(config_path='../3DGS-GPT/config', config_name='vocabulary', version_base='1.2')
def main(config):
    dataset = GaussianDataset(config)
    dataset.prepare_data(0)

if __name__ == '__main__':
    print('This is data')
    main()
