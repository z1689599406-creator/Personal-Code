# ---------- Self-Attention Layer ----------
class SelfAttention2D(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, in_channels)
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        # reshape to (B, heads, C//heads, H*W)
        q = q.view(B, self.num_heads, C // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)

#       q.transpose(-2, -1): 维度变为 $(B, \text{heads}, N, d_k)$
#       矩阵乘法: (B, \text{heads}, N, d_k) \times (B, \text{heads}, d_k, N) \rightarrow 结果 attn: $(B, \text{heads}, N, N)$
#       物理意义: 这是一个“像素对像素”的相关性地图
#       N*N代表了图中任意两个像素点之间的关系强度。
        attn = torch.softmax(torch.matmul(q.transpose(-2, -1), k) / math.sqrt(C // self.num_heads), dim=-1)
        out = torch.matmul(attn, v.transpose(-2, -1)).transpose(-2, -1)
        out = out.contiguous().view(B, C, H, W)
        out = self.proj_out(out)
        return x + out  # 残差连接确保了注意力模块只是在为原始图像特征添加“全局上下文”作为补充，极大地增强了训练的稳定性