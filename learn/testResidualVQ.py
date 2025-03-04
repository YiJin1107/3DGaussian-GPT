import torch
from vector_quantize_pytorch import ResidualVQ
 
# 初始化向量量化器
vq = ResidualVQ(
    dim=256, # 输入向量维度
    num_quantizers=2, # 量化层深度
    codebook_size=512,  # 码本大小
    decay=0.8,         # 指数移动平均衰减
    commitment_weight=1, # 承诺损失权重
    shared_codebook=True, # 共享码本
)
 
# 生成随机输入数据
x = torch.randn(1, 1024, 256)
 
# 进行量化
quantized, indices, commit_loss = vq(x)
 
print(quantized.shape)  # 输出: (1, 1024, 256)
print(indices.shape)    # 输出: (1, 1024)
print(commit_loss)      # 输出: 1