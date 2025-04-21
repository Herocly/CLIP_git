import torch
import torch.nn as nn
from dataset import disease_text_dict,diseases,


class SelfAttentionFuser(nn.Module):
    def __init__(self, input_dim, hidden_dim=None):
        # 输入特征维度和隐藏特征维度
        super().__init__()
        if hidden_dim is None: hidden_dim = input_dim
        self.W_q = nn.Linear(input_dim, hidden_dim)  # Query变换
        self.W_k = nn.Linear(input_dim, hidden_dim)  # Key变换
        self.W_v = nn.Linear(input_dim, hidden_dim)  # Value变换
        self.output_proj = nn.Linear(hidden_dim, input_dim)  # 输出投影

    def forward(self, descs):  # descs: [N, D]
        Q = self.W_q(descs)  # [N, H]
        K = self.W_k(descs)  # [N, H]
        V = self.W_v(descs)  # [N, H]
        attn_scores = torch.matmul(Q, K.t()) / Q.shape[-1] ** 0.5  # [N, N]
        attn_weights = f.softmax(attn_scores, dim=-1)  # [N, N], 行归一化
        out = torch.matmul(attn_weights, V)  # [N, H]
        out = self.output_proj(out)  # [N, D]

        fused = out.mean(dim=0)  # [D]
        return fused, attn_weights  # 方便分析可返回权重
