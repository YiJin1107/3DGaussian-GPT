import warnings
warnings.filterwarnings("ignore")
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import pytorch_lightning as pl
import hydra
from lightning_utilities.core.rank_zero import rank_zero_only
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from vector_quantize_pytorch import ResidualVQ


from dataset.Gaussian import GaussianDataset,point_collate_fn
from model.PTv3encoder import Ptv3Encoder
from model.decoder import resnet34_decoder
from trainer import create_trainer, step, create_conv_batch
import yaml
from pathlib import Path

class GaussTokenizationPTv3(pl.LightningModule):
    def __init__(self,config,test_mode=False):
        super().__init__()
        self.config = config
        self.test_mode = test_mode
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.skip_quant = config.skip_quant
        self.ce_output = config.ce_output

        # 加载类别配置文件
        category_config_path = Path('./config/bin') / f'{config.category}.yaml'
        with open(category_config_path, 'r') as f:
            category_config = yaml.safe_load(f)

        self.bin_config = {
            'coord': {'num_bins': config.coord_bins, 'min': category_config['coord']['min'], 'max': category_config['coord']['max']},      
            'color': {'num_bins': config.color_bins, 'min': category_config['color']['min'], 'max': category_config['color']['max']},        
            'opacity': {'num_bins': config.opacity_bins, 'min': category_config['opacity']['min'], 'max': category_config['opacity']['max']},        
            'scale': {'num_bins': config.scale_bins, 'min': category_config['scale']['min'], 'max': category_config['scale']['max']},  
            'rot': {'num_bins': config.rot_bins, 'min': category_config['rot']['min'], 'max': category_config['rot']['max']}           
        }
        
        # hyper
        self.register_buffer('smoothing_weight', 
            torch.tensor([2, 10, 200, 10, 2], dtype=torch.float32, device=self.device).view(1, 1, -1))
        # dataset
        self.train_dataset = GaussianDataset(config=self.config,bin_config=self.bin_config)
        self.val_dataset = GaussianDataset(config=self.config,bin_config=self.bin_config,split='test')
        # encoder
        self.encoder = Ptv3Encoder(config)
        # vq
        self.pre_quant = torch.nn.Linear(config.enc_dim, config.embed_dim)
        self.vq = ResidualVQ(
            dim=self.config.embed_dim, # 嵌入向量的维度
            codebook_size=self.config.n_embed,  # codebook size 总的嵌入数量
            num_quantizers=config.embed_levels, # 量化层深度
            commitment_weight=self.config.embed_loss_weight,  # 控制输入输出的匹配程度
            stochastic_sample_codes=True,
            sample_codebook_temp=config.stochasticity,  # 采样随机性
            shared_codebook=self.config.embed_share, # 共享码本
            decay=self.config.code_decay, # 控制码字的更新速率
            rotation_trick = True, # 
            kmeans_init = True, # kmeans初始化
            kmeans_iters = 5000, # 
            threshold_ema_dead_code = 2, # 命中率阈值
            use_cosine_sim = True , # 余弦相似度度量距离
            )

        self.post_quant = torch.nn.Linear(config.embed_dim, config.enc_dim)
        # pooling
        self.cluster_conv = torch.nn.Conv1d(
            in_channels=config.enc_dim,
            out_channels=config.enc_dim,
            kernel_size=config.cluster_size,
            stride=config.cluster_size,
            padding=0,
            groups=1  # 保持全连接模式以学习更灵活的下采样
        )
        
        self.extend_deconv = torch.nn.ConvTranspose1d(
            in_channels=config.enc_dim,
            out_channels=config.enc_dim,
            kernel_size=config.cluster_size,
            stride=config.cluster_size,
            padding=0,
            output_padding=0  # 根据实际需要调整
        )

        # decoder
        self.decoder = resnet34_decoder(config.enc_dim,config.num_tokens,config.ce_output)


    def configure_optimizers(self):
        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.cluster_conv.parameters()) + list(self.extend_deconv.parameters())
        parameters += list(self.pre_quant.parameters()) 
        parameters += list(self.post_quant.parameters())        
        parameters += list(self.vq.parameters())        


        if self.skip_quant:
            for param in self.pre_quant.parameters():
                param.requires_grad = False
            for param in self.post_quant.parameters():
                param.requires_grad = False
            for param in self.vq.parameters():
                param.requires_grad = False
        else:
            # for param in self.encoder.parameters():
            #     param.requires_grad = False
            # for param in self.decoder.parameters():
            #     param.requires_grad = False
            # for param in self.cluster_conv.parameters():
            #     param.requires_grad = False
            # for param in self.extend_deconv.parameters():
            #     param.requires_grad = False
            pass
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
        if self.test_mode:
            return  # 在测试模式下跳过训练步骤
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        scheduler.step()  # type: ignore
        point = self.encoder(data)
        feat = point['feat'][point['serialized_order'][0]] # (N,D)
        # feat = self.bn(feat)
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

                starget = torch.nn.functional.conv1d(
                    torch.nn.functional.one_hot(
                        y.long().reshape(-1), num_classes=self.config.num_tokens).float().unsqueeze(1), 
                        self.smoothing_weight, bias=None, stride=1, padding=2, dilation=1, groups=1) # (N * 14,1,C)
                starget_skip_feat = torch.nn.functional.normalize(starget.reshape(-1, logits.shape[-2]*logits.shape[-1]), p=1.0, dim=-1,eps=1e-12).squeeze(1)
                starget_per_feat = torch.nn.functional.normalize(starget, p=1.0, dim=-1,eps=1e-12).squeeze(1) # (N * 14,C)
                loss_skip_feat = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-2]*logits.shape[-1]),starget_skip_feat,reduction='mean') # (N,14*C)
                loss_per_feat = torch.nn.functional.cross_entropy(logits.reshape(-1, self.config.num_tokens),starget_per_feat,reduction='mean') # (N * 14,C)
                loss = loss_per_feat + loss_skip_feat * 0.1 + commit_loss
            else: 
                decoded_x = decoded_x_conv.reshape(-1, decoded_x_conv.shape[-1])[conv_mask, :]
                y = data['feat'][point['serialized_order'][0]] # (N,14)
                loss = torch.nn.functional.mse_loss(decoded_x, y, reduction='mean') + commit_loss
        else:
            if self.ce_output:
                y = data['target'][point['serialized_order'][0]] # (N,14)
                logits = decoded_x_conv.reshape(-1, decoded_x_conv.shape[-2], decoded_x_conv.shape[-1])[conv_mask, :, :]
                starget = torch.nn.functional.conv1d(
                    torch.nn.functional.one_hot(
                        y.long().reshape(-1), num_classes=self.config.num_tokens).float().unsqueeze(1), 
                        self.smoothing_weight, bias=None, stride=1, padding=2, dilation=1, groups=1) # (N * 14,1,C)
                starget_skip_feat = torch.nn.functional.normalize(starget.reshape(-1, logits.shape[-2]*logits.shape[-1]), p=1.0, dim=-1,eps=1e-12).squeeze(1)
                starget_per_feat = torch.nn.functional.normalize(starget, p=1.0, dim=-1,eps=1e-12).squeeze(1) # (N * 14,C)
                loss_skip_feat = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-2]*logits.shape[-1]),starget_skip_feat,reduction='mean') # (N,14*C)
                loss_per_feat = torch.nn.functional.cross_entropy(logits.reshape(-1, self.config.num_tokens),starget_per_feat,reduction='mean') # (N * 14,C)
                loss = loss_per_feat + loss_skip_feat * 0.1
            else:
                decoded_x = decoded_x_conv.reshape(-1, decoded_x_conv.shape[-1])[conv_mask, :]
                y = data['feat'][point['serialized_order'][0]] # (N,14)
                loss = torch.nn.functional.mse_loss(decoded_x, y, reduction='mean')
        
        loss = loss / self.config.gradient_accumulation_steps  # scale the loss to account for gradient accumulation
        self.manual_backward(loss)
        # accumulate gradients of `n` batches
        if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
            if self.skip_quant:
                params = [self.encoder, self.decoder, self.cluster_conv, self.extend_deconv]
            else:
                params = [self.encoder, self.cluster_conv, self.pre_quant, self.post_quant, self.extend_deconv, self.decoder,self.vq]

            step(optimizer, params)
            optimizer.zero_grad(set_to_none=True)  # type: ignore

            # 手动释放不再需要的变量
            del feat, cluster_batches, encoded_x_conv, decoded_x_conv, logits, starget
            torch.cuda.empty_cache()  # 清空CUDA缓存

        self.log("train/loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        if not self.skip_quant:
            self.log("train/vq_loss", commit_loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("lr", optimizer.param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)  # type: ignore

    def validation_step(self, data, batch_idx, dataloader_idx=0):
        point = self.encoder(data)
        feat = point['feat'][point['serialized_order'][0]] # (N,D)
        # feat = self.bn(feat)
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
                starget = torch.nn.functional.conv1d(
                    torch.nn.functional.one_hot(
                        y.long().reshape(-1), num_classes=self.config.num_tokens).float().unsqueeze(1), 
                        self.smoothing_weight, bias=None, stride=1, padding=2, dilation=1, groups=1) # (N * 14,1,C)
                starget = torch.nn.functional.normalize(starget, p=1.0, dim=-1,eps=1e-12).squeeze(1) # (N * 14,C)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, self.config.num_tokens),starget,reduction='mean') + commit_loss
            else: 
                decoded_x = decoded_x_conv.reshape(-1, decoded_x_conv.shape[-1])[conv_mask, :]
                y = data['feat'][point['serialized_order'][0]] # (N,14)
                loss = torch.nn.functional.mse_loss(decoded_x, y, reduction='mean') + commit_loss
        else:
            if self.ce_output:
                y = data['target'][point['serialized_order'][0]] # (N,14)
                logits = decoded_x_conv.reshape(-1, decoded_x_conv.shape[-2], decoded_x_conv.shape[-1])[conv_mask, :, :]
                starget = torch.nn.functional.conv1d(
                    torch.nn.functional.one_hot(
                        y.long().reshape(-1), num_classes=self.config.num_tokens).float().unsqueeze(1), 
                        self.smoothing_weight, bias=None, stride=1, padding=2, dilation=1, groups=1) # (N * 14,1,C)
                starget = torch.nn.functional.normalize(starget, p=1.0, dim=-1,eps=1e-12).squeeze(1)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, self.config.num_tokens),starget,reduction='mean')
            else:
                decoded_x = decoded_x_conv.reshape(-1, decoded_x_conv.shape[-1])[conv_mask, :]
                y = data['feat'][point['serialized_order'][0]] # (N,14)
                loss = torch.nn.functional.mse_loss(decoded_x, y, reduction='mean')


        if not torch.isnan(loss).any():
            self.log(f"val/loss", loss.item(), add_dataloader_idx=False, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        if not self.skip_quant:
            self.log(f"val/vq_loss", commit_loss.item(), add_dataloader_idx=False, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        if self.current_epoch % self.config.save_epoch == 0:
            if self.ce_output:
                class_indices = torch.argmax(logits, dim=-1).long().detach().cpu() # (N,14,C) -> (N,14)
                decoded_x = self.reconstruct_from_discrete(class_indices)
                yy = self.reconstruct_from_discrete(y.detach().cpu())

            del feat, cluster_batches, encoded_x_conv, decoded_x_conv, logits, starget
            torch.cuda.empty_cache()

            self._save_gaussian_ply(decoded_x, point['batch'].detach().cpu(), batch_idx,'val')
            self._save_gaussian_ply(yy, point['batch'].detach().cpu(), batch_idx,'o')
            self._save_chamfer_distance(decoded_x,yy,point['batch'].detach().cpu(),batch_idx)

    @rank_zero_only
    def _save_chamfer_distance(self, decoded_x, target_x, batch_indices, batch_idx):
        import os
        from torch import cdist
        import numpy as np
        
        # 创建保存目录
        save_dir = Path("runs") / self.config.experiment / "CD"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 分离不同样本的点云
        unique_batches = torch.unique(batch_indices)
        for bid in unique_batches:
            mask = (batch_indices == bid)
            decoded_points = decoded_x[mask]
            target_points = target_x[mask]

            # 计算倒角距离
            dist_matrix = cdist(decoded_points, target_points)
            chamfer_dist = dist_matrix.min(dim=1)[0].mean() + dist_matrix.min(dim=0)[0].mean()

            # 保存结果到.txt文件
            with open(save_dir / f'batch_{batch_idx}_sample_{bid.item()}_cd.txt', 'w') as f:
                f.write(f"Chamfer Distance: {chamfer_dist.item()}\n")

    @rank_zero_only
    def _save_gaussian_ply(self, decoded_x, batch_indices, batch_idx,test):
        import os
        from plyfile import PlyData, PlyElement
        import numpy as np
        
        # 创建保存目录
        save_dir = Path("runs") / self.config.experiment / "3DGS" / f"epoch_{self.current_epoch}{test}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 分离不同样本的点云
        unique_batches = torch.unique(batch_indices)
        for bid in unique_batches:
            mask = (batch_indices == bid)
            processed_feat = decoded_x[mask]

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

            
            # 保存文件
            ply = PlyData([PlyElement.describe(vertex_elements, 'vertex')])
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
            shuffle=False, 
            drop_last=False, 
            num_workers=self.config.num_workers, 
            pin_memory=True,
            collate_fn=point_collate_fn
            )
        return loader
    
    def clustering(self, feat, batch):
        """使用可学习的1D卷积进行聚类(自动填充保证整除)"""
        max_index = batch.max().item() + 1
        cluster_feats = []
        cluster_batches = []

        for i in range(max_index):
            sample_mask = (batch == i)
            sample_feat = feat[sample_mask]  # (N_i, D)
            
            # 计算需要填充的长度
            remainder = sample_feat.size(0) % self.config.cluster_size
            if remainder != 0:
                padding = self.config.cluster_size - remainder
                sample_feat = sample_feat.unsqueeze(0)  # (1, N_i, D)
                sample_feat = torch.nn.functional.pad(sample_feat, (0, 0, 0, padding), "replicate")
                sample_feat = sample_feat.squeeze(0)  # (N_i + padding, D)
            # 添加通道维度并转置为 (D, N_i)
            sample_feat = sample_feat.T.unsqueeze(0)  # (1, D, N_i)
            
            # 应用可学习卷积
            clustered = self.cluster_conv(sample_feat)  # (1, D, C)
            
            # 转置回 (C, D) 并收集结果
            cluster_feat = clustered.squeeze(0).T  # (C, D)
            cluster_feats.append(cluster_feat)
            cluster_batches.append(torch.full((cluster_feat.size(0),), i, device=feat.device))

        return torch.cat(cluster_feats), torch.cat(cluster_batches)

    def extend(self, feat, cluster_batches, batches):
        """使用反卷积进行特征扩展（自动裁剪到原始点数）"""
        max_index = cluster_batches.max().item() + 1
        extended_feats = []

        for i in range(max_index):
            # 获取当前样本的簇特征
            cluster_mask = (cluster_batches == i)
            sample_feat = feat[cluster_mask]  # (C, D)
            
            # 添加通道维度并转置为 (D, C)
            sample_feat = sample_feat.T.unsqueeze(0)  # (1, D, C)
            
            # 应用反卷积
            extended = self.extend_deconv(sample_feat)  # (1, D, C*S)
            
            # 转置回 (C*S, D)
            extended_feat = extended.squeeze(0).T  # (C*S, D)
            
            # 根据原始点数裁剪（考虑可能的填充）
            original_points = (batches == i).sum().item()
            extended_feats.append(extended_feat[:original_points])

        return torch.cat(extended_feats, dim=0)

    def create_conv_batch(self, encoded_features, batch, batch_size):
        return create_conv_batch(encoded_features, batch, batch_size, self.device)
    
    def reconstruct_from_discrete(self, feat):
        """从离散索引重建原始数据（近似值）"""
        # 获取离散特征和分箱配置
        discrete_feat = feat
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

        return reconstructed

    def load_state_dict(self, state_dict, strict=True):
        # 过滤掉缺失的bn参数
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('bn.')}
        # 过滤掉vq参数
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('vq.')}
        super().load_state_dict(state_dict, strict=False)

    def on_load_checkpoint(self, checkpoint):
        # 删除所有optimizer相关的状态
        # if 'optimizer_states' in checkpoint:
        #     del checkpoint['optimizer_states']
        # if 'lr_schedulers' in checkpoint:
        #     del checkpoint['lr_schedulers']
        pass

# 使用Hydra库的装饰器 配置并管理python参数
@hydra.main(config_path='./config', config_name='gaussiangpt_test', version_base='1.2')
def main(config):
    trainer = create_trainer("3DGaussTokens", config)
    model = GaussTokenizationPTv3(config,True)
    resume = './runs/car数据集前300epoch-wovq/checkpoints/299-0.ckpt'
    trainer.fit(model,ckpt_path=resume)


if __name__ == '__main__':
    main()
