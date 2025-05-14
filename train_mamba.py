import random

import omegaconf
import trimesh
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import pytorch_lightning as pl
import hydra
from easydict import EasyDict
from lightning_utilities.core.rank_zero import rank_zero_only
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from dataset.quantized_soup import QuantizedSoupCreator
from dataset.Gaussian import GaussianWithSequenceIndices,point_collate_fn

from model.mamba import QuantSoupMamba,QuantSoupMambaConfig

from trainer import create_trainer, step, get_rvqvae_v0_decoder
from util.misc import accuracy
from util.visualization import plot_vertices_and_faces
from util.misc import get_parameters_from_state_dict
import yaml
from pathlib import Path
import time  # 导入 time 模块

class QuantSoupModelTrainer(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vq_cfg = omegaconf.OmegaConf.load(Path(config.vq_resume).parents[1] / "config.yaml")
        self.save_hyperparameters()

        # bin_config
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

        # dataset
        self.train_dataset = GaussianWithSequenceIndices(config,self.bin_config,'train')
        self.val_dataset = GaussianWithSequenceIndices(config,self.bin_config,'val')
        print("Dataset Lengths:", len(self.train_dataset), len(self.val_dataset))
        print("Batch Size:", self.config.batch_size)
        print("Dataloader Lengths:", len(self.train_dataset) // self.config.batch_size, len(self.val_dataset) // self.config.batch_size)
        
        
        # model
        self.sequencer = QuantizedSoupCreator(self.config, self.vq_cfg)
        self.sequencer.freeze_vq()

        self.model_cfg = get_qsoup_model_config(config, self.vq_cfg)
        self.model = QuantSoupMamba(self.model_cfg, self.vq_cfg)

        self.output_dir_image = Path(f'runs/{self.config.experiment}/image')
        self.output_dir_image.mkdir(exist_ok=True, parents=True)
        self.output_dir_mesh = Path(f'runs/{self.config.experiment}/mesh')
        self.output_dir_mesh.mkdir(exist_ok=True, parents=True)

        if self.config.ft_resume is not None:
            self.model.load_state_dict(get_parameters_from_state_dict(torch.load(self.config.ft_resume, map_location='cpu')['state_dict'], "model"))
        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer = self.model.configure_optimizers(
            self.config.weight_decay, self.config.lr,
            (self.config.beta1, self.config.beta2), 'cuda'
        )
        max_steps = int(self.config.max_epoch * len(self.train_dataset) / self.config.batch_size / 2)
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
        if self.config.force_lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.config.force_lr
        # (B,t:N*D)
        sequence_in, sequence_out, pfin, pfout, cluster_batches,batch = self.sequencer(data, data['js'])

        logits, loss = self.model(sequence_in, pfin, pfout, self.sequencer, targets=sequence_out)

        acc = accuracy(logits.detach(), sequence_out, ignore_label=2, device=self.device)
        self.log("train/ce_loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/acc", acc.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        loss = loss / self.config.gradient_accumulation_steps  # scale the loss to account for gradient accumulation
        self.manual_backward(loss)
        # accumulate gradients of `n` batches
        if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
            step(optimizer, [self.model])
            optimizer.zero_grad(set_to_none=True)  # type: ignore
        self.log("lr", optimizer.param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)  # type: ignore

    def validation_step(self, data, batch_idx):
        sequence_in, sequence_out, pfin, pfout, cluster_batches,batch = self.sequencer(data,data['js'])
        logits, loss = self.model(sequence_in, pfin, pfout, self.sequencer, targets=sequence_out)
        acc = accuracy(logits.detach(), sequence_out, ignore_label=2, device=self.device)
        if not torch.isnan(loss).any():
            self.log("val/ce_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        if not torch.isnan(acc).any():
            self.log("val/acc", acc.item(), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

    @rank_zero_only
    def on_validation_epoch_end(self):
        decoder = get_rvqvae_v0_decoder(self.vq_cfg, self.config.vq_resume, self.device)
        for k in range(self.config.num_val_samples):
            data_dict = self.val_dataset.get(random.randint(0, len(self.val_dataset) - 1))
            data_dict = {k: v.to(self.device) for k, v in data_dict.items()}
            print('开始生成...')
            start_time = time.time()  # 记录开始时间
            soup_sequence, in_idx, out_idx, target, cluster_batches, batch = self.sequencer.get_completion_sequence(
                data_dict,
                12
            )
            
            y = self.model.generate(
                soup_sequence, in_idx, out_idx, self.sequencer,self.model_cfg.finemb_size + 3,self.config.max_val_tokens,eos_token_id=1,
            )
            end_time = time.time()  # 记录结束时间
            print(f'生成完成，耗时: {end_time - start_time:.2f} 秒')  # 输出耗时

            # class_indices = self.sequencer.decode(target, decoder)
            # gaussians = self.reconstruct_from_discrete(class_indices)
            # self._save_gaussian_ply(gaussians,'o')


            if y is None:
                continue
            
            class_indices = self.sequencer.decode(y[0], decoder)
            gaussians = self.reconstruct_from_discrete(class_indices)
            self._save_gaussian_ply(gaussians,'r')


            
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

    @rank_zero_only
    def _save_gaussian_ply(self, decoded_x, flag):
        import os
        from plyfile import PlyData, PlyElement
        import numpy as np
        
        # 创建保存目录
        save_dir = Path("runs") / self.config.experiment / "3DGS" / f"epoch_{self.current_epoch}"
        save_dir.mkdir(parents=True, exist_ok=True)
        print('Gaussians output',save_dir)
        processed_feat = decoded_x

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
        ply.write(str(save_dir / f'val_{flag}.ply'))

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
    

def get_qsoup_model_config(config, vq_config): 
    cfg = QuantSoupMambaConfig(
        d_model = config.mamba.d_model, # 隐藏层维度
        n_layer = config.mamba.n_layer, # block层数
        vocab_size = config.mamba.vocab_size, # 词典大小
        ssm_cfg = dict(layer=config.mamba.ssm_cfg), # Mamba模型选择
        rms_norm = config.mamba.rms_norm,
        residual_in_fp32 = config.mamba.residual_in_fp32,
        fused_add_norm = config.mamba.fused_add_norm,
        pad_vocab_size_multiple = config.mamba.pad_vocab_size_multiple,
        finemb_size = vq_config.embed_levels, # 量化层数量
        foutemb_size = config.max_len, # 3DGaussian数量
    )

    return cfg


@hydra.main(config_path='./config', config_name='gaussiangpt', version_base='1.2')
def main(config):
    trainer = create_trainer("3DGaussianSoup", config)
    model = QuantSoupModelTrainer(config)
    resume = './runs/car生成前160epoch/checkpoints/159-0.ckpt'
    trainer.fit(model, ckpt_path=resume)



if __name__ == '__main__':
    main()
