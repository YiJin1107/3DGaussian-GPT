import torch
from collections.abc import Sequence

class Collect(object):
    def __init__(self, keys, offset_keys_dict=None, **kwargs):
        """
        e.g. Collect(keys=[coord], feat_keys=[coord, color])
        """
        if offset_keys_dict is None:
            offset_keys_dict = dict(offset="coord")
        self.keys = keys
        self.offset_keys = offset_keys_dict
        self.kwargs = kwargs

    def __call__(self, data_dict):
        data = dict()
        if isinstance(self.keys, str):
            self.keys = [self.keys]
        for key in self.keys:
            data[key] = data_dict[key]
        for key, value in self.offset_keys.items():
            data[key] = torch.tensor([data_dict[value].shape[0]])
        for name, keys in self.kwargs.items():
            name = name.replace("_keys", "")
            assert isinstance(keys, Sequence)
            data[name] = torch.cat([data_dict[key].float() for key in keys], dim=1)
        return data

data_dict = {
    "coord": torch.rand(100, 3),  # 100个点的坐标
    "color": torch.rand(100, 3)    # 100个点的颜色
}

# 假设这是你的输入数据字典
data_dict = {
    "coord": torch.rand(100, 3),  # 100个点的坐标
    "color": torch.rand(100, 3),   # 100个点的颜色
    "segment": torch.randint(0, 5, (100,)),  # 100个点的分段信息
}

# 创建 Collect 类的实例
collect_transform = Collect(keys=["coord", "color"], offset_keys_dict={"offset": "coord"}, feat_keys=["color"])

# 调用 Collect 实例来处理数据字典
collected_data = collect_transform(data_dict)

# 输出处理后的数据
print(collected_data)
