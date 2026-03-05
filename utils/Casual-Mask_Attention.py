import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalMask(nn.Module):
    """
    通用的独立因果掩码模块
    用于自回归模型，确保模型在位置 t 只能看到 1 到 t 的信息。
    """
    def __init__(self, block_size):
        super().__init__()
        # 创建下三角矩阵并调整形状以匹配多头注意力的维度
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        # 注册为 buffer，使其随模型保存且能在 GPU/CPU 间自动转移，但不参与梯度更新
        self.register_buffer("mask", mask)

    def forward(self, att):
        """
        Args:
            att: 原始的注意力分数矩阵，形状为 (B, nh, T, T)
        """
        T = att.size(-1)
        # 将 mask 中为 0 的位置（即未来的信息）填充为负无穷大
        return att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))


class CausalSelfAttention(nn.Module):
    """
    完整的多头因果自注意力层
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # 将 key, query, value 的投影合并在一个 Linear 层中以提高 GPU 计算效率
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 输出投影层
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # 正则化 Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # 【修改点】引入我们刚才独立出来的掩码模块，不再将 mask 称为 bias
        self.causal_mask_module = CausalMask(config.block_size)
        
        # 检测当前 PyTorch 版本是否支持 Flash Attention (PyTorch >= 2.0)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x):
        B, T, C = x.size() # Batch size, Sequence length, Embedding dimensionality

        # 1. 一次性计算所有的 query, key, value
        qkv = self.c_attn(x)
        # 沿特征维度切分为三份
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # 2. 调整形状并转置，使得 head 维度靠前: (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 3. 计算注意力并混合信息
        if self.flash:
            # 开启 Flash Attention 硬件加速
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            # 手动实现标准注意力机制: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            
            # 【修改点】调用解耦出来的掩码模块，截断未来信息
            att = self.causal_mask_module(att)
            
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            
            # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = att @ v 

        # 4. 将所有 head 的输出重新拼接在一起
        # transpose 变回 (B, T, nh, hs)，contiguous 保证内存连续，view 展平为 (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 5. 最后进行一次线性投影和残差 Dropout
        y = self.resid_dropout(self.c_proj(y))
        return y