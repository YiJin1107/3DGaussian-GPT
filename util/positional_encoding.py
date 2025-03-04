import torch
import torch.nn as nn

# 嵌入类，用于生成位置编码的嵌入表示
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        # 创建嵌入函数
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = [] # 嵌入函数列表
        d = self.kwargs['input_dims']
        out_dim = 0
        # 通过恒等函数将数据本身与位置编码连接
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x) # 将匿名函数添加 这里是恒等函数
            out_dim += d
        
        # 计算频率带
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']: # 对数采样 以指数方式增长 适合频率变化大的情况
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else: # 线性采样 均匀增长 适合频率变化小的情况
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        # 添加周期函数
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim
    
    # 将所有嵌入函数的输出连接到一起
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

# 获取嵌入器函数
def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos], # 用于生成位置编码
    }
    # embed接受输入数据本身x
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim
