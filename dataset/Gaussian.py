from torch.utils.data import Dataset
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
import omegaconf
import yaml
import math

class GaussianDataset(Dataset):
    def __init__(self,config,bin_config,split='train',max_sh_degree=3):
        super(GaussianDataset, self).__init__()
        self.split = split
        self.data_root = Path('data',config.dataset,config.category,split)
        self.data_list = self.get_data_list(self.data_root)
        self.discretize = config.ce_output
        self.config = config
        self.max_rotation = config.max_rotation  # 最大旋转角度（度）
        self.scale_range = eval(config.scale_range)  # 缩放范围
        self.color_jitter = config.color_jitter  # 颜色扰动强度
        self.density_dropout = config.density_dropout  # 密度丢弃概率
        

        self.bin_config = bin_config

        self.data_cache = [self._load_single_data(path) for path in self.data_list]

    def prepare_data(self,idx):
        data_dict = self.get_data(idx)
        data_dict = self.transform_data(data_dict)


        if self.discretize:
            data_dict = self._discretize_gaussian(data_dict)
            # if True:
            #     data_dict = self.reconstruct_from_discrete(data_dict)
            #     self._save_gaussian_ply(data_dict,idx)
            #     exit(-1)

        feat = data_dict['feat']
        feat_min = feat.min(dim=0, keepdim=True)[0]
        feat_max = feat.max(dim=0, keepdim=True)[0]
        data_dict['feat'] = (feat - feat_min) / (feat_max - feat_min + 1e-6)  # 防止除以0
        

        return data_dict

    def __getitem__(self, idx):
        return self.prepare_data(idx)

    def __len__(self):
        return len(self.data_list)

    def get_data_list(self,data_root):
        data_list = list(data_root.glob('*'))  
        return data_list

    def _load_single_data(self, path):
        """加载单个PLY文件数据"""
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
        
        return dict(coord=coord, feat=features)
    
    def get_data(self, idx):
        """直接从内存中获取数据"""
        return self.data_cache[idx]
    
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
        # 初始化离散特征张量
        discrete_feat = torch.zeros(feat.shape[0], 14, dtype=torch.long)
        # 遍历每个特征值进行离散化
        for i, (key, config) in enumerate(bin_config.items()):
            if key == 'coord':
                start, end = 0, 3
            elif key == 'color':
                start, end = 3, 6
            elif key == 'opacity':
                start, end = 6, 7
            elif key == 'scale':
                start, end = 7, 10
            elif key == 'rot':
                start, end = 10, 14

            # 对每个特征值生成分箱边界
            for j in range(start, end):
                bins = torch.linspace(config['min'][j - start], config['max'][j - start], config['num_bins'])
                
                # 对特征值进行分箱
                indices = torch.bucketize(feat[:, j], bins) - 1
                indices = indices.clamp(0, config['num_bins']-1)
                
                # 将离散化结果存入离散特征张量
                discrete_feat[:, j] = indices

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

        # 遍历每个特征值进行重建
        for i, (key, config) in enumerate(bin_config.items()):
            if key == 'coord':
                start, end = 0, 3
            elif key == 'color':
                start, end = 3, 6
            elif key == 'opacity':
                start, end = 6, 7
            elif key == 'scale':
                start, end = 7, 10
            elif key == 'rot':
                start, end = 10, 14

            # 对每个特征值生成分箱边界并进行线性插值
            for j in range(start, end):
                bins = torch.linspace(config['min'][j - start], config['max'][j - start], config['num_bins'] + 1)
                # 计算每个离散索引对应的插值位置
                indices = discrete_feat[:, j].clamp(0, len(bins)-2)  # 确保索引在有效范围内
                alpha = (discrete_feat[:, j] - indices.float())  # 插值权重
                # 线性插值
                reconstructed[:, j] = (1 - alpha) * bins[indices] + alpha * bins[indices + 1]

        data_dict['feat'] = reconstructed
        return data_dict

class GaussianWithSequenceIndices(GaussianDataset):
    def __init__(self,config,bin_config,split='train'):
        super().__init__(config=config,bin_config=bin_config,split=split)
        resume = './runs/car数据集前340epoch/config.yaml'
        vq_cfg = omegaconf.OmegaConf.load(resume)
        self.vq_depth = vq_cfg.embed_levels
        self.block_size = config.block_size
        print('vq_depth:',self.vq_depth)
        print('block_size:',self.block_size)
        max_inner_face_len = 0
        self.padding = int(config.padding * self.block_size)
        self.sequence_stride = config.sequence_stride

        # length
        self.sequence_indices = []
        max_face_sequence_len = 0
        min_face_sequence_len = 1e7

        for i in range(len(self.data_cache)):
            sequence_len = math.ceil(len(self.data_cache[i]['feat']) / config.cluster_size) * self.vq_depth + 1 + 1
            max_face_sequence_len = max(max_face_sequence_len, sequence_len)
            min_face_sequence_len = min(min_face_sequence_len, sequence_len)
            self.sequence_indices.append((i, 0, False))
            # 处理序列长度大于码本大小的情况
            for j in range(config.sequence_stride, max(1, sequence_len - self.block_size + self.padding + 1), config.sequence_stride):  # todo: possible bug? +1 added recently
                self.sequence_indices.append((i, j, True if split == 'train' else False))
            if sequence_len > self.block_size: 
                self.sequence_indices.append((i, sequence_len - self.block_size, False))
        print('Length of', split, len(self.sequence_indices))
        print('Shortest Gaussian sequence', min_face_sequence_len)
        print('Longest Gaussian sequence', max_face_sequence_len)

    def __len__(self):
        return len(self.sequence_indices)
    
    def __getitem__(self, idx):
        i, j, randomness = self.sequence_indices[idx] # i:Batch idx j:sequence start
        if randomness:
            sequence_len = math.ceil(len(self.data_cache[i]['feat']) / self.config.cluster_size) * self.vq_depth + 1 + 1
            j = min(max(0, j + np.random.randint(-self.sequence_stride // 2, self.sequence_stride // 2)), sequence_len - self.block_size + self.padding)
        data_dict = self.prepare_data(i)
        data_dict['js'] = torch.tensor([j]).long()
        return data_dict
    
    def get(self,idx):
        return self.__getitem__(idx)
    
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

@hydra.main(config_path='../config', config_name='gaussiangpt', version_base='1.2')
def main(config):

    category_config_path = Path('./config/bin') / f'{config.category}.yaml'
    with open(category_config_path, 'r') as f:
        category_config = yaml.safe_load(f)

    bin_config = {
        'coord': {'num_bins': config.coord_bins, 'min': category_config['coord']['min'], 'max': category_config['coord']['max']},      
        'color': {'num_bins': config.color_bins, 'min': category_config['color']['min'], 'max': category_config['color']['max']},        
        'opacity': {'num_bins': config.opacity_bins, 'min': category_config['opacity']['min'], 'max': category_config['opacity']['max']},        
        'scale': {'num_bins': config.scale_bins, 'min': category_config['scale']['min'], 'max': category_config['scale']['max']},  
        'rot': {'num_bins': config.rot_bins, 'min': category_config['rot']['min'], 'max': category_config['rot']['max']}           
    }

    dataset = GaussianWithSequenceIndices(config=config,bin_config=bin_config)
    for i in range(1000):
        data_dict = dataset[i]
        if data_dict['js'] > 0:
            print(data_dict['js'])
            break

if __name__ == '__main__':
    print('This is data')
    main()
