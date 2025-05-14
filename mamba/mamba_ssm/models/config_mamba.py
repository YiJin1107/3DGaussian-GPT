from dataclasses import dataclass, field

'''
存储Mamba模型的配置信息的数据类,包括模型参数和配置字段
'''
@dataclass
class MambaConfig:

    d_model: int = 2560
    d_intermediate: int = 0
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8 # 词典大小的约数
    tie_embeddings: bool = True
