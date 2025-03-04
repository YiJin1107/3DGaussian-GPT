import torch
import torch.nn as nn
import torch_geometric
import torch_scatter  

from util.positional_encoding import get_embedder


class GraphEncoder(nn.Module):

    def __init__(self, no_max_pool=True, aggr='mean', graph_conv="edge", use_point_features=False, output_dim=512):
        super().__init__()
        self.no_max_pool = no_max_pool
        self.use_point_features = use_point_features
        # 获取嵌入器 以及其维度
        self.embedder, self.embed_dim = get_embedder(10)
        self.conv = graph_conv
        # 初始化图卷积层
        self.gc1 = get_conv(self.conv, self.embed_dim * 3 + 7, 64, aggr=aggr)
        self.gc2 = get_conv(self.conv, 64, 128, aggr=aggr)
        self.gc3 = get_conv(self.conv, 128, 256, aggr=aggr)
        self.gc4 = get_conv(self.conv, 256, 256, aggr=aggr)
        self.gc5 = get_conv(self.conv, 256, output_dim, aggr=aggr)
        
        # 初始化归一化层
        self.norm1 = torch_geometric.nn.BatchNorm(64)
        self.norm2 = torch_geometric.nn.BatchNorm(128)
        self.norm3 = torch_geometric.nn.BatchNorm(256)
        self.norm4 = torch_geometric.nn.BatchNorm(256)

        self.relu = nn.ReLU()
    # x : (N,) batch : (N,) edge_index : (2,E)
    def forward(self, x, edge_index, batch):
        # 单独拿出 坐标 进行位置信息嵌入
        x_0 = self.embedder(x[:, :3])
        x_1 = self.embedder(x[:, 3:6])
        x_2 = self.embedder(x[:, 6:9])
        x_n = x[:, 9:12] # 法线
        x_ar = x[:, 12:13] # 面积
        # 角度
        x_an_0 = x[:, 13:14]
        x_an_1 = x[:, 14:15]
        x_an_2 = x[:, 15:]
        # 重新将一个三角的所有特征连接到一起
        x = torch.cat([x_0, x_1, x_2, x_n, x_ar, x_an_0, x_an_1, x_an_2], dim=-1)
        x = self.relu(self.norm1(self.gc1(x, edge_index)))
        x = self.norm2(self.gc2(x, edge_index))
        point_features = x
        x = self.relu(x)
        x = self.relu(self.norm3(self.gc3(x, edge_index)))
        x = self.relu(self.norm4(self.gc4(x, edge_index)))
        x = self.gc5(x, edge_index)
        if not self.no_max_pool:
            x = torch_scatter.scatter_max(x, batch, dim=0)[0]
            x = x[batch, :] # 重新索引保持形状(N,512)
        if self.use_point_features: # 与原始点的特征相连
            return torch.cat([x, point_features], dim=-1)
        return x


class GraphEncoderTriangleSoup(nn.Module):

    def __init__(self, aggr='mean', graph_conv="edge"):
        super().__init__()
        self.embedder, self.embed_dim = get_embedder(10)
        self.conv = graph_conv
        self.gc1 = get_conv(self.conv, self.embed_dim * 3 + 7, 96, aggr=aggr)
        self.gc2 = get_conv(self.conv, 96, 192, aggr=aggr)
        self.gc3 = get_conv(self.conv, 192, 384, aggr=aggr)
        self.gc4 = get_conv(self.conv, 384, 384, aggr=aggr)
        self.gc5 = get_conv(self.conv, 384, 576, aggr=aggr)

        self.norm1 = torch_geometric.nn.BatchNorm(96)
        self.norm2 = torch_geometric.nn.BatchNorm(192)
        self.norm3 = torch_geometric.nn.BatchNorm(384)
        self.norm4 = torch_geometric.nn.BatchNorm(384)

        self.relu = nn.ReLU()

    @staticmethod
    def distribute_features(features, face_indices, num_vertices):
        N, F = features.shape
        features = features.reshape(N * 3, F // 3)
        assert features.shape[0] == face_indices.shape[0] * face_indices.shape[1], "Features and face indices must match in size"
        vertex_features = torch.zeros([num_vertices, features.shape[1]], device=features.device)
        torch_scatter.scatter_mean(features, face_indices.reshape(-1), out=vertex_features, dim=0)
        distributed_features = vertex_features[face_indices.reshape(-1), :]
        distributed_features = distributed_features.reshape(N, F)
        return distributed_features

    def forward(self, x, edge_index, faces, num_vertices):
        x_0 = self.embedder(x[:, :3])
        x_1 = self.embedder(x[:, 3:6])
        x_2 = self.embedder(x[:, 6:9])
        x = torch.cat([x_0, x_1, x_2, x[:, 9:]], dim=-1)
        x = self.relu(self.norm1(self.gc1(x, edge_index)))
        x = self.distribute_features(x, faces, num_vertices)
        x = self.relu(self.norm2(self.gc2(x, edge_index)))
        x = self.distribute_features(x, faces, num_vertices)
        x = self.relu(self.norm3(self.gc3(x, edge_index)))
        x = self.distribute_features(x, faces, num_vertices)
        x = self.relu(self.norm4(self.gc4(x, edge_index)))
        x = self.distribute_features(x, faces, num_vertices)
        x = self.gc5(x, edge_index)
        x = self.distribute_features(x, faces, num_vertices)
        return x


def get_conv(conv, in_dim, out_dim, aggr):
    if conv == 'sage':
        return torch_geometric.nn.SAGEConv(in_dim, out_dim, aggr=aggr)
    elif conv == 'gat':
        return torch_geometric.nn.GATv2Conv(in_dim, out_dim, fill_value=aggr)
    elif conv == 'edge':
        return torch_geometric.nn.EdgeConv(
            torch.nn.Sequential(
                torch.nn.Linear(in_dim * 2, 2 * out_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(2 * out_dim, out_dim),
            ),
            aggr=aggr,
        )
