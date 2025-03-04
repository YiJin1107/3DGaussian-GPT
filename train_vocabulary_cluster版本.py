import warnings
warnings.filterwarnings("ignore")
import omegaconf
import trimesh
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import pytorch_lightning as pl
import hydra
from lightning_utilities.core.rank_zero import rank_zero_only
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from vector_quantize_pytorch import ResidualVQ

from dataset.Gaussian import GaussianDataset,point_collate_fn
from model.softargmax import softargmax
from model.PTv3encoder import Ptv3Encoder
from model.decoder import resnet34_decoder
from trainer import create_trainer, step, create_conv_batch


class GaussTokenizationPTv3(pl.LightningModule):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.skip_quant = config.skip_quant
        self.ce_output = config.ce_output
        
        self.bin_config = {
            'coord': {'num_bins': config.coord_bins, 'min_max': (-0.5, 0.5)},      
            'color': {'num_bins': config.color_bins, 'min_max': (-3, 6)},        
            'opacity': {'num_bins': config.opacity_bins, 'min_max': (-6, 12)},        
            'scale': {'num_bins': config.scale_bins, 'min_max': (-17, -2)},        
            'rot': {'num_bins': config.rot_bins, 'min_max': (-2, 4)}           
        }
        self.register_buffer('smoothing_weight', 
            torch.tensor([2, 10, 200, 10, 2], dtype=torch.float32, device=self.device).view(1, 1, -1))
        # dataset
        self.train_dataset = GaussianDataset(config=self.config)
        self.val_dataset = GaussianDataset(config=self.config,split='val')
        # encoder
        self.encoder = Ptv3Encoder(config)
        # RVQ
        if not self.skip_quant:
            self.pre_quant = torch.nn.Linear(config.dec_dim, config.embed_dim)
            self.vq = ResidualVQ(
            dim=self.config.embed_dim, # 嵌入向量的维度
            codebook_size=self.config.n_embed,  # codebook size 总的嵌入数量
            num_quantizers=config.embed_levels, # 量化层深度
            commitment_weight=self.config.embed_loss_weight,  # 控制输入输出的匹配程度
            stochastic_sample_codes=True,
            sample_codebook_temp=config.stochasticity,  # 采样随机性
            shared_codebook=self.config.embed_share, # 共享码本
            decay=self.config.code_decay, # 控制码字的更新速率
            )
            self.post_quant = torch.nn.Linear(config.embed_dim, config.dec_dim)
        else:
            # 当跳过量化时，不创建这些层
            self.pre_quant = None
            self.vq = None
            self.post_quant = None

        # decoder
        self.decoder = resnet34_decoder(config.dec_dim,config.num_tokens,config.ce_output)

    def configure_optimizers(self):
        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        if not self.skip_quant:
            parameters += list(self.pre_quant.parameters()) 
            parameters += list(self.post_quant.parameters())
            parameters += list(self.vq.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=self.config.lr, amsgrad=True, weight_decay=self.config.weight_decay)
        max_steps = int(self.config.max_epoch * len(self.train_dataset) / self.config.batch_size)
        print('Max Steps | First cycle:', max_steps)
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer, first_cycle_steps=max_steps, cycle_mult=1.0,
            max_lr=self.config.lr, min_lr=self.config.min_lr,
            warmup_steps=self.config.warmup_steps, gamma=1.0
        )
        return [optimizer], [scheduler]

    def training_step(self, data, batch_idx):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        scheduler.step()  # type: ignore
        point = self.encoder(data)
        feat = point['feat'][point['serialized_order'][0]] # (N,D)
        feat,cluster_batches = self.clustering(feat,point['batch']) # (Np = N/cluster_size,D)
        
        if not self.skip_quant:
            feat = self.pre_quant(feat) # (Np,E)
            feat, _, commit_loss = self.vq(feat.unsqueeze(0)) # (1,Np,E)
            feat = feat.squeeze(0)
            commit_loss = commit_loss.mean()
            feat = self.post_quant(feat) #(Np,D)
        else:
            feat = feat  # 保持原特征维度

        feat = self.extend(feat,cluster_batches,point['batch']) # (N,D)
        encoded_x_conv, conv_mask = self.create_conv_batch(feat,point['batch'] , self.config.batch_size) # （B,D,mN)
        decoded_x_conv = self.decoder(encoded_x_conv) # (B,mN,14,numtokens)

        if not self.skip_quant:
            if self.ce_output:
                y = data['target'][point['serialized_order'][0]] # (N,14)
                logits = decoded_x_conv.reshape(-1, decoded_x_conv.shape[-2], decoded_x_conv.shape[-1])[conv_mask, :, :] # (N,14,C)
                otarget = torch.nn.functional.one_hot(y.long().reshape(-1), num_classes=self.config.num_tokens).float()  # (N * 14,C)
                otarget = otarget.unsqueeze(1) # (N * 14,1,C)
                starget = torch.nn.functional.conv1d(otarget, self.smoothing_weight, bias=None, stride=1, padding=2, dilation=1, groups=1) # (N * 14,1,C)
                starget = torch.nn.functional.normalize(starget, p=1.0, dim=-1,eps=1e-12).squeeze(1) # (N * 14,C)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, self.config.num_tokens),starget,reduction='mean') + commit_loss
            else: 
                y = data['feat'][point['serialized_order'][0]] # (N,14)
                loss = torch.nn.functional.mse_loss(decoded_x, y, reduction='mean') + commit_loss
        else:
            if self.ce_output:
                y = data['target'][point['serialized_order'][0]] # (N,14)
                logits = decoded_x_conv.reshape(-1, decoded_x_conv.shape[-2], decoded_x_conv.shape[-1])[conv_mask, :, :]
                otarget = torch.nn.functional.one_hot(y.long().reshape(-1), num_classes=self.config.num_tokens).float()
                otarget = otarget.unsqueeze(1)
                starget = torch.nn.functional.conv1d(otarget, self.smoothing_weight, bias=None, stride=1, padding=2, dilation=1, groups=1)
                starget = torch.nn.functional.normalize(starget, p=1.0, dim=-1,eps=1e-12).squeeze(1)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, self.config.num_tokens),starget,reduction='mean')
            else:
                y = data['feat'][point['serialized_order'][0]] # (N,14)
                loss = torch.nn.functional.mse_loss(decoded_x, y, reduction='mean')
        
        loss = loss / self.config.gradient_accumulation_steps  # scale the loss to account for gradient accumulation
        self.manual_backward(loss)
        # accumulate gradients of `n` batches
        if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
            params = [self.encoder, self.decoder]
            if not self.skip_quant:
                params += [self.pre_quant, self.post_quant]
            step(optimizer, params)
            optimizer.zero_grad(set_to_none=True)  # type: ignore
        self.log("train/loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        if not self.skip_quant:
            self.log("train/vq_loss", commit_loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("lr", optimizer.param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)  # type: ignore

    def validation_step(self, data, batch_idx, dataloader_idx=0):
        point = self.encoder(data)
        feat = point['feat'][point['serialized_order'][0]] # (N,D)
        feat,cluster_batches = self.clustering(feat,point['batch']) # (Np = N/cluster_size,D)
        
        if not self.skip_quant:
            feat = self.pre_quant(feat) # (Np,E)
            feat, _, commit_loss = self.vq(feat.unsqueeze(0)) # (1,Np,E)
            feat = feat.squeeze(0)
            commit_loss = commit_loss.mean()
            feat = self.post_quant(feat) #(Np,D)
        else:  # 跳过量化层时直接使用原始特征
            feat = feat  # 保持原特征维度

        feat = self.extend(feat,cluster_batches,point['batch']) # (N,D)
        encoded_x_conv, conv_mask = self.create_conv_batch(feat,point['batch'] , self.config.batch_size)
        decoded_x_conv = self.decoder(encoded_x_conv) # (1,N,14,C)

        if not self.skip_quant:
            if self.ce_output:
                y = data['target'][point['serialized_order'][0]] # (N,14)
                logits = decoded_x_conv.reshape(-1, decoded_x_conv.shape[-2], decoded_x_conv.shape[-1])[conv_mask, :, :] # (N,14,C)
                otarget = torch.nn.functional.one_hot(y.long().reshape(-1), num_classes=self.config.num_tokens).float()  # (N * 14,C)
                otarget = otarget.unsqueeze(1) # (N * 14,1,C)
                starget = torch.nn.functional.conv1d(otarget, self.smoothing_weight, bias=None, stride=1, padding=2, dilation=1, groups=1) # (N * 14,1,C)
                starget = torch.nn.functional.normalize(starget, p=1.0, dim=-1,eps=1e-12).squeeze(1) # (N * 14,C)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, self.config.num_tokens),starget,reduction='mean') + commit_loss
            else: 
                y = data['feat'][point['serialized_order'][0]] # (N,14)
                loss = torch.nn.functional.mse_loss(decoded_x, y, reduction='mean') + commit_loss
        else:
            if self.ce_output:
                y = data['target'][point['serialized_order'][0]] # (N,14)
                logits = decoded_x_conv.reshape(-1, decoded_x_conv.shape[-2], decoded_x_conv.shape[-1])[conv_mask, :, :]
                otarget = torch.nn.functional.one_hot(y.long().reshape(-1), num_classes=self.config.num_tokens).float()
                otarget = otarget.unsqueeze(1)
                starget = torch.nn.functional.conv1d(otarget, self.smoothing_weight, bias=None, stride=1, padding=2, dilation=1, groups=1)
                starget = torch.nn.functional.normalize(starget, p=1.0, dim=-1,eps=1e-12).squeeze(1)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, self.config.num_tokens),starget,reduction='mean')
            else:
                y = data['feat'][point['serialized_order'][0]] # (N,14)
                loss = torch.nn.functional.mse_loss(decoded_x, y, reduction='mean')


        if not torch.isnan(loss).any():
            self.log(f"val/loss", loss.item(), add_dataloader_idx=False, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        if not self.skip_quant:
            self.log(f"val/vq_loss", commit_loss.item(), add_dataloader_idx=False, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        if self.current_epoch % 10 == 0 and self.current_epoch != 0:
            if self.ce_output:
                class_indices = torch.argmax(logits, dim=-1).long()
                decoded_x = self.reconstruct_from_discrete(class_indices)

            self._save_gaussian_ply(decoded_x, point['batch'], batch_idx)
        

    @rank_zero_only
    def _save_gaussian_ply(self, decoded_x, batch_indices, batch_idx):
        import os
        from plyfile import PlyData, PlyElement
        import numpy as np
        
        # 创建保存目录
        save_dir = Path("runs") / self.config.experiment / "3DGS" / f"epoch_{self.current_epoch}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 分离不同样本的点云
        unique_batches = torch.unique(batch_indices)
        for bid in unique_batches:
            mask = (batch_indices == bid)
            sample_data = decoded_x[mask].detach().cpu().numpy()
            
            coord = sample_data[:, :3]          # x,y,z 
            features_dc = sample_data[:, 3:6]   # f_dc_0, f_dc_1, f_dc_2
            opacities = sample_data[:, 6]       # opacity
            scales = sample_data[:, 7:10]       # scale_0, scale_1, scale_2
            rots = sample_data[:, 10:14]         # rot_0~rot_3
            
            vertex_data = np.zeros(len(coord), dtype=[
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
                ('opacity', 'f4'),
                ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
                ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
            ])
            
            vertex_data['x'] = coord[:,0]
            vertex_data['y'] = coord[:,1]
            vertex_data['z'] = coord[:,2]
            vertex_data['f_dc_0'] = features_dc[:,0]
            vertex_data['f_dc_1'] = features_dc[:,1]
            vertex_data['f_dc_2'] = features_dc[:,2]
            vertex_data['opacity'] = opacities
            vertex_data['scale_0'] = scales[:,0]
            vertex_data['scale_1'] = scales[:,1]
            vertex_data['scale_2'] = scales[:,2]
            vertex_data['rot_0'] = rots[:,0]
            vertex_data['rot_1'] = rots[:,1]
            vertex_data['rot_2'] = rots[:,2]
            vertex_data['rot_3'] = rots[:,3]
            
            # 保存文件
            ply = PlyData([PlyElement.describe(vertex_data, 'vertex')])
            ply.write(str(save_dir / f'batch_{batch_idx}_sample_{bid.item()}.ply'))

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            drop_last=True, 
            num_workers=self.config.num_workers, 
            pin_memory=True,
            collate_fn=point_collate_fn
            )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            drop_last=True, 
            num_workers=self.config.num_workers, 
            pin_memory=True,
            collate_fn=point_collate_fn
            )
        return loader
    
    def clustering(self, feat, batch):
        cluster_size = self.config.cluster_size
        max_index = batch.max().item() + 1
        cluster_feats = []
        cluster_batches = []

        for i in range(max_index):
            sample_mask = (batch == i)
            sample_feat = feat[sample_mask]
            
            # 分块处理
            num_points = sample_feat.size(0)
            num_clusters = (num_points + cluster_size - 1) // cluster_size  # 向上取整
            
            # 按cluster_size分块求平均
            clusters = []
            for c in range(num_clusters):
                start = c * cluster_size
                end = min((c+1)*cluster_size, num_points)
                cluster_mean = sample_feat[start:end].mean(dim=0)
                clusters.append(cluster_mean)
            
            cluster_feats.append(torch.stack(clusters))
            cluster_batches.append(torch.full((len(clusters),), i, device=feat.device))

        return torch.cat(cluster_feats), torch.cat(cluster_batches)


    def extend(self, feat, cluster_batches, batches):
        # 输入feat形状：(Np, D)，cluster_batches形状：(Np,)
        # 输出形状应与原始点云数量一致：(N, D)
        cluster_size = self.config.cluster_size
        device = feat.device
        
        # 生成相对位置编码（簇内位置）
        position = torch.arange(cluster_size, device=device).float()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.config.dec_dim, 2, device=device).float() / self.config.dec_dim))
        pos_enc = torch.zeros(cluster_size, self.config.dec_dim, device=device)
        pos_enc[:, 0::2] = torch.sin(position.unsqueeze(-1) * inv_freq.unsqueeze(0))
        pos_enc[:, 1::2] = torch.cos(position.unsqueeze(-1) * inv_freq.unsqueeze(0))
        pos_enc = pos_enc * self.config.pos_enc_weight  # 可配置的位置编码强度
        
        # 按样本扩展特征
        max_index = cluster_batches.max().item() + 1
        extended_feats = []
        
        for i in range(max_index):
            # 获取当前样本的所有簇
            cluster_mask = (cluster_batches == i)
            sample_feat = feat[cluster_mask]
            
            # 每个簇扩展为cluster_size个点
            expanded_feat = sample_feat.unsqueeze(1).repeat(1, cluster_size, 1)  # (C, S, D)
            expanded_feat += pos_enc.unsqueeze(0)  # 添加位置编码
            expanded_feat = expanded_feat.view(-1, self.config.dec_dim)  # (C*S, D)
            
            # 根据原始点云数量裁剪（处理最后一个不完整的簇）
            original_points = (batches == i).sum().item()
            expanded_feat = expanded_feat[:original_points]
            
            extended_feats.append(expanded_feat)
            
        return torch.cat(extended_feats, dim=0) 

    def create_conv_batch(self, encoded_features, batch, batch_size):
        return create_conv_batch(encoded_features, batch, batch_size, self.device)
    
    def reconstruct_from_discrete(self, feat):
        """从离散索引重建原始数据（近似值）"""
        # 获取离散特征和分箱配置
        discrete_feat = feat
        bin_config = self.bin_config
        
        # 初始化重建后的特征张量
        reconstructed = torch.zeros(discrete_feat.shape[0], 14, dtype=torch.float32,device=self.device)
        # 坐标重建 (前3维)
        coord_bins = torch.linspace(*bin_config['coord']['min_max'], bin_config['coord']['num_bins'] + 1,device=self.device)
        coord_values = (coord_bins[1:] + coord_bins[:-1]) / 2  # 取分箱中点
        reconstructed[:, :3] = coord_values[discrete_feat[:, :3].clamp(0, len(coord_values)-1)]

        # 颜色特征重建 (3-6维)
        color_bins = torch.linspace(*bin_config['color']['min_max'], bin_config['color']['num_bins'] + 1,device=self.device)
        color_values = (color_bins[1:] + color_bins[:-1]) / 2
        reconstructed[:, 3:6] = color_values[discrete_feat[:, 3:6].clamp(0, len(color_values)-1)]

        # 透明度重建 (第6维)
        opacity_bins = torch.linspace(*bin_config['opacity']['min_max'], bin_config['opacity']['num_bins'] + 1,device=self.device)
        opacity_values = (opacity_bins[1:] + opacity_bins[:-1]) / 2
        reconstructed[:, 6] = opacity_values[discrete_feat[:, 6].clamp(0, len(opacity_values)-1)]

        # 缩放系数重建 (7-9维)
        scale_bins = torch.linspace(*bin_config['scale']['min_max'], bin_config['scale']['num_bins'] + 1,device=self.device)
        scale_values = (scale_bins[1:] + scale_bins[:-1]) / 2
        reconstructed[:, 7:10] = scale_values[discrete_feat[:, 7:10].clamp(0, len(scale_values)-1)]

        # 旋转参数重建 (10-13维)
        rot_bins = torch.linspace(*bin_config['rot']['min_max'], bin_config['rot']['num_bins'] + 1,device=self.device)
        rot_values = (rot_bins[1:] + rot_bins[:-1]) / 2
        reconstructed[:, 10:14] = rot_values[discrete_feat[:, 10:14].clamp(0, len(rot_values)-1)]
        
        return reconstructed

# 使用Hydra库的装饰器 配置并管理python参数
@hydra.main(config_path='../3DGS-GPT/config', config_name='vocabulary', version_base='1.2')
def main(config):
    trainer = create_trainer("3DGaussTokens", config)
    model = GaussTokenizationPTv3(config)
    resume = '../3DGS-GPT/runs/02141504_3DGaussTokens_ex1_ragged-identity/checkpoints/09-0.ckpt'
    trainer.fit(model, ckpt_path=resume)


if __name__ == '__main__':
    main()
