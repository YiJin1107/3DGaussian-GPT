import torch
import torch.nn as nn
from model.mamba_base import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from dataclasses import dataclass, field

class QuantSoupMamba(MambaLMHeadModel):

    def __init__(self, config, vq_config):
        super().__init__(config) # MambaConfig
        self.padding_idx = 2
        self.tokens_per_gs = config.finemb_size
        self.finemb_size = 3 + config.finemb_size  
        self.foutemb_size = 3 + config.foutemb_size  
        vocab_size = vq_config.n_embed + 1 + 1 + 1  # +1 for start, +1 for stop, +1 for pad 16384 + 3
        self.vocab_size = vocab_size
        print('Model Vocab Size:', vocab_size)
        print('Model Padding Index:', self.padding_idx)
        print('Model Fin Size:', self.finemb_size)
        print('Model Fout Size:', self.foutemb_size)
        print('Model hidden Size:',config.d_model)
        print('Model layer Num:',config.n_layer)
        
        self.input_layer = nn.Linear(vq_config.embed_dim, config.d_model)
        # self.wpe_emmbeds = nn.Embedding(config.block_size, config.d_model) # 位置编码
        self.extra_embeds = nn.Embedding(3, config.d_model, padding_idx=self.padding_idx)  # 处理其他标识符 start,end,pad
        self.fin_embeds = nn.Embedding(self.finemb_size, config.d_model, padding_idx=self.padding_idx) # 每个gs对应的几个token顺序索引的嵌入
        self.fout_embeds = nn.Embedding(self.foutemb_size, config.d_model, padding_idx=self.padding_idx) # 每个token对应的gs顺序索引的嵌入
        
        # self.tie_weights()

    def tie_weights(self):
        '''
            将语言模型头与词嵌入的权重进行绑定
            权重绑定的思想是, 在语言模型中, 输入的词嵌入和输出的词概率分布应该有一定的关联。
        '''
        if self.config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

    def forward(self, idx, fin, fout, tokenizer, targets=None, inference_params=None,num_last_tokens=0,position_ids=None):
        device = idx.device

        b, t = idx.size()  # (B,t)
        
        # 处理输入嵌入
        embed = torch.zeros((b * t, self.config.d_model), dtype=torch.float32, device=device)  # (B*t,d_model)
        idx_in_extra = torch.isin(idx, torch.LongTensor([0, 1, 2]).to(device)).reshape(-1)  # (B*t,)
        idx_flat = idx.reshape(-1)  # (B*t,)
        # 处理其他索引
        embed[idx_in_extra, :] = self.extra_embeds(idx_flat[idx_in_extra])  # (extra,d_model)
        # 从码本中获取向量，直接用作嵌入
        embed[~idx_in_extra, :] = self.input_layer(tokenizer.embed(idx_flat[~idx_in_extra] - 3))  # (b*t - extra,d_model)
        tok_emb = embed.reshape(b, t, -1)  # token embeddings of shape (b, t, d_model)
        
        # 处理 fin 和 fout 嵌入
        fin_emb = self.fin_embeds(fin)  # face inner embeddings of shape (b,t, d_model)
        fout_emb = self.fout_embeds(fout)  # face outer embeddings of shape (b,t, d_model)

        # 合并嵌入
        sum_emb = tok_emb + fin_emb + fout_emb
        
        # 通过 Mamba 主干网络
        hidden_states = self.backbone(sum_emb, inference_params=inference_params)

        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
            
        # 计算 logits
        logits = self.lm_head(hidden_states)
        
        if targets is not None:
            # 如果提供了目标，计算损失
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.padding_idx)
        else:
            # 推理时只返回 logits
            loss = None
        return logits, loss
    
@dataclass
class QuantSoupMambaConfig(MambaConfig):
    finemb_size:int = 2
    foutemb_size:int = 10000