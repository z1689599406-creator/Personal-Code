import torch
from torch.utils.data import Dataset, DataLoader

# 1. 定义 Dataset：告诉程序“数据在哪”和“怎么取一个”
class MyDataset(Dataset):
    def __init__(self, data_tensor, target_tensor):
        # 存下所有数据
        self.data = data_tensor
        self.target = target_tensor

    def __len__(self):
        # 告诉程序总共有多少个样本
        return len(self.data)

    def __getitem__(self, index):
        # 核心：根据索引 index 返回“一个”样本及其标签
        return self.data[index], self.target[index]

# 2. 模拟一些数据 (例如 100 个样本，每个特征维度是 10)
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))

# 3. 实例化并交给 DataLoader
dataset = MyDataset(X, y)
# DataLoader 负责打乱(shuffle)、分批(batch)和多线程加速(num_workers)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 4. 测试使用
for batch_x, batch_y in dataloader:
    print(batch_x.shape) # 输出 torch.Size([16, 10])
    print(len(batch_x))