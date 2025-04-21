import torch
import torch.utils.data
from dataset import get_shifted_sequence
from trainer import get_rvqvae_v0_encoder_vq,create_conv_batch

class QuantizedSoupCreator(torch.nn.Module):
    def __init__(self, config, vq_cfg):
        super().__init__()
        self.vq_cfg = vq_cfg
        self.vq_depth = self.vq_cfg.embed_levels
        self.vq_is_shared = self.vq_cfg.embed_share
        self.vq_num_codes_per_level = self.vq_cfg.n_embed
        self.vq_dim = self.vq_cfg.embed_dim
        assert config.num_tokens == self.vq_cfg.num_tokens, "Number of tokens must match"
        self.block_size = config.block_size
        self.start_sequence_token = 0
        self.end_sequence_token = 1
        self.pad_face_token = 2
        self.num_tokens = config.num_tokens - 3
        self.padding = int(config.padding * self.block_size)
        self.rq_transformer_input = False
        self.encoder, self.pre_quant, self.post_quant, self.cluster_conv, self.extend_deconv,self.vq = self.get_rvq_encoders(config.vq_resume)
        self.cluster_size = config.cluster_size

    def get_rvq_encoders(self, resume):
        '''
            功能:重新加载编码器和量化器
        '''
        return get_rvqvae_v0_encoder_vq(self.vq_cfg, resume)

    def freeze_vq(self):
        '''
            功能:冻结编码器和量化器
        '''
        for model in [self.encoder, self.pre_quant, self.post_quant, self.vq]:
            for param in model.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def embed(self, idx):
        '''
            功能:根据索引从码本中取出向量
            idx:(b*t,)
        '''
        assert self.vq_is_shared, "Only shared embedding is supported"
        all_codes = self.vq.codebooks[0][idx].reshape(-1, self.vq_dim) # (b*t,v_dim)
        return all_codes

    @torch.no_grad()
    def get_indices(self, data_dict):
        '''
            功能:获取排序好的码本索引序列,以及卷积核大小
        '''        
        point = self.encoder(data_dict)
        batch = point['batch']
        feat = point['feat'][point['serialized_order'][0]] # (N,D)
        feat,cluster_batches = self.clustering(feat,batch) # (Np = N/cluster_size,D)
        feat = self.pre_quant(feat) # (Np,E)
        _, all_indices,_ = self.vq(feat.unsqueeze(0)) # (1,Np,E)
        all_indices = all_indices.squeeze(0) # (Np,D)
        return all_indices,batch,cluster_batches


    @torch.no_grad()
    def forward(self, data_dict, js, force_full_sequence=False):
        for model in [self.encoder, self.pre_quant, self.post_quant, self.cluster_conv, self.extend_deconv, self.vq]:
            model.eval()

        all_indices,batch,cluster_batches = self.get_indices(data_dict) # (N,D)
        batch_size = js.shape[0]
        sequences = []
        targets = []
        position_inners = []
        position_outers = []
        max_sequence_length_x = 0
        for k in range(batch_size):
            sequence_k = all_indices[cluster_batches == k, :]
            # (N,D) 每个GS内部量化嵌入的顺序
            inner_face_id_k = torch.arange(0, self.vq_depth , device=js.device).reshape(1, -1).expand(sequence_k.shape[0], -1)
            # (N,D) 每个量化嵌入所属GS的顺序
            outer_face_id_k = torch.arange(0, sequence_k.shape[0], device=js.device).reshape(-1, 1).expand(-1, self.vq_depth)
            
            # 整体偏移 0,1,2用作其他表示
            sequence_k = sequence_k.reshape(-1) + 3 # (N*D,)
            inner_face_id_k = inner_face_id_k.reshape(-1) + 3 # (N*D,)
            outer_face_id_k = outer_face_id_k.reshape(-1) + 3 # (N*D,)

            # 添加开始token和结束token
            prefix = [torch.tensor([self.start_sequence_token], device=js.device)] 
            postfix = [torch.tensor([self.end_sequence_token], device=js.device)]
            sequence_k = torch.cat(prefix + [sequence_k] + postfix).long()
            inner_face_id_k = torch.cat(prefix + [inner_face_id_k] + postfix).long()
            outer_face_id_k = torch.cat(prefix + [outer_face_id_k] + postfix).long()
            
            # 截断最大处理长度 or 填充不足长度
            j = js[k]
            if force_full_sequence:
                end_index = len(sequence_k)
            else:
                end_index = min(j + self.block_size, len(sequence_k))
            x_in = sequence_k[j:end_index]
            y_in = sequence_k[j + 1:end_index + 1] # y是x的下一时刻的序列
            fpi_in = inner_face_id_k[j:end_index]
            fpo_in = outer_face_id_k[j:end_index].cpu()

            max_sequence_length_x = max(max_sequence_length_x, len(x_in))
            pad_len_x = self.block_size - len(x_in)
            x_pad = torch.tensor([self.pad_face_token for _ in range(0, pad_len_x)], device=js.device)

            pad_len_y = len(x_in) + len(x_pad) - len(y_in)
            pad_len_fpi = self.block_size - len(fpi_in)
            pad_len_fpo = self.block_size - len(fpo_in)


            y_pad = torch.tensor([self.pad_face_token for _ in range(0, pad_len_y)], device=js.device)
            fpi_in_pad = torch.tensor([self.pad_face_token for _ in range(0, pad_len_fpi)], device=js.device)
            fpo_in_pad = torch.tensor([self.pad_face_token for _ in range(0, pad_len_fpo)])

            x = torch.cat((x_in, x_pad)).long()
            y = torch.cat((y_in, y_pad)).long()
            fpi = torch.cat((fpi_in, fpi_in_pad)).long()
            fpo = torch.from_numpy(get_shifted_sequence(torch.cat((fpo_in, fpo_in_pad)).numpy())).long().to(js.device)

            sequences.append(x) # (N*D,)
            targets.append(y)
            position_inners.append(fpi)
            position_outers.append(fpo)
        print('max_sequence_length_x:',max_sequence_length_x)
        sequences = torch.stack(sequences, dim=0)[:, :max_sequence_length_x].contiguous() # (B,N) N:Gaussians num * Depth
        targets = torch.stack(targets, dim=0)[:, :max_sequence_length_x].contiguous() # (B,N)
        position_inners = torch.stack(position_inners, dim=0)[:, :max_sequence_length_x].contiguous() # (B,N)
        position_outers = torch.stack(position_outers, dim=0)[:, :max_sequence_length_x].contiguous() # (B,N)
        return sequences, targets, position_inners, position_outers,cluster_batches,batch

    @torch.no_grad()
    def clustering(self, feat, batch):
            """使用可学习的1D卷积进行聚类(自动填充保证整除)"""
            max_index = batch.max().item() + 1
            cluster_feats = []
            cluster_batches = []

            for i in range(max_index):
                sample_mask = (batch == i)
                sample_feat = feat[sample_mask]  # (N_i, D)
                
                # 计算需要填充的长度
                remainder = sample_feat.size(0) % self.cluster_size
                if remainder != 0:
                    padding = self.cluster_size - remainder
                    sample_feat = torch.nn.functional.pad(sample_feat, (0,0,0,padding), "constant", 0)
                
                # 添加通道维度并转置为 (D, N_i)
                sample_feat = sample_feat.T.unsqueeze(0)  # (1, D, N_i)
                
                # 应用可学习卷积
                clustered = self.cluster_conv(sample_feat)  # (1, D, C)
                
                # 转置回 (C, D) 并收集结果
                cluster_feat = clustered.squeeze(0).T  # (C, D)
                cluster_feats.append(cluster_feat)
                cluster_batches.append(torch.full((cluster_feat.size(0),), i, device=feat.device))

            return torch.cat(cluster_feats), torch.cat(cluster_batches)

    @torch.no_grad()
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
    
    @torch.no_grad()
    def get_completion_sequence(self, data_dict, tokens):

        soup_sequence, target, in_idx, out_idx, cluster_batches,batch = self.forward(
            data_dict=data_dict,
            js=torch.zeros([1], device=data_dict['js'].device).long(),
            force_full_sequence=True
        )

        soup_sequence = soup_sequence[0] # (B,max_N,D) ->(max_N,D)
        in_idx = in_idx[0]
        out_idx = out_idx[0]
        target = target[0]
        if isinstance(tokens, int):
            num_pre_tokens = tokens
        else:
            num_pre_tokens = int(len(target) * tokens)
        x = (
            soup_sequence[:num_pre_tokens][None, ...], # (1,num_pre_tokens,D)
            in_idx[:num_pre_tokens][None, ...],
            out_idx[:num_pre_tokens][None, ...],
            target[None, ...],
            cluster_batches,
            batch,
        )
        return x

    def encode_sequence(self, sequence):
        '''
            功能:根据索引从码本中找到对应向量
        '''
        N = sequence.shape[0]
        E, D = self.vq_dim, self.vq_depth
        all_codes = self.vq.get_codes_from_indices(sequence).permute(1, 2, 0)
        encoded_x = all_codes.reshape(N, E, D).sum(-1) # 在D维度上相加
        return encoded_x

    @torch.no_grad()
    def decode(self, sequence, decoder,cluster_batches,batch):
        mask = torch.isin(sequence, torch.tensor([self.start_sequence_token, self.end_sequence_token, self.pad_face_token], device=sequence.device)).logical_not()
        sequence = sequence[mask]
        sequence = sequence - 3
        sequence_len = (sequence.shape[0] // (self.vq_depth)) * (self.vq_depth)
        sequence = sequence[:sequence_len].reshape(-1, self.vq_depth) # (N,D)
        encoded_x = self.encode_sequence(sequence) # (N,E)
        encoded_x = self.post_quant(encoded_x) # (N,D)
        feat = self.extend(encoded_x,cluster_batches,batch) # (N,D)
        encoded_x_conv, conv_mask = self.create_conv_batch(feat,batch,1)
        decoded_x_conv = decoder(encoded_x_conv) # (1,N,14,C)
        logits = decoded_x_conv.reshape(-1, decoded_x_conv.shape[-2], decoded_x_conv.shape[-1])[conv_mask, :, :] # (N,14,C)
        class_indices = torch.argmax(logits, dim=-1).long().detach().cpu()
 
        return class_indices # 返回类别索引

    def create_conv_batch(self, encoded_features, batch, batch_size):
        return create_conv_batch(encoded_features, batch, batch_size, encoded_features.device)