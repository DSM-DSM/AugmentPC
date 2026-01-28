import math
import torch
import torch.nn as nn
import torch.optim


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Args:
        timesteps: [B,] 时间步（整数或浮点数）
        dim: 嵌入维度
        max_period: 控制频率范围（默认 10000）
    Returns:
        [B, dim] 时间编码
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]  # [B, half]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, 2*half]
    if dim % 2:  # 如果 dim 是奇数，补零
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class MLPDiffusion(nn.Module):
    def __init__(self, dim_in, unique_label_num = 0, task_type='reg', is_y_cond=False, dim_t=128,
                 dim_hidden_layers=[256, 256], dropout=0.1):
        super().__init__()
        self.dim_t = dim_t
        self.task_type = task_type
        self.is_y_cond = is_y_cond
        self.unique_label_num = unique_label_num

        # === 1. 时间编码模块 ===
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

        # === 2. 条件输入 y 的嵌入（可选）===
        if self.is_y_cond:
            if self.task_type == 'reg':
                self.label_emb = nn.Linear(1, dim_t)  # 回归任务
            elif self.unique_label_num > 0:
                self.label_emb = nn.Embedding(self.unique_label_num, dim_t)  # 分类任务

        # === 3. 数据投影层 ===
        self.proj = nn.Linear(dim_in, dim_t)

        # === 4. MLP 主体 ===
        layers = []
        input_dim = dim_t
        for d in dim_hidden_layers:
            layers.append(nn.Linear(input_dim, d))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            input_dim = d
        layers.append(nn.Linear(input_dim, dim_in))  # 输出层
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, timesteps, y=None):
        """
        Args:
            x: [B, d_in] 输入数据
            timesteps: [B,] 时间步（整数或浮点数）
            y: [B,] 条件输入（可选）
        Returns:
            [B, d_in] 预测值
        """
        # 1. 时间编码
        t_emb = timestep_embedding(timesteps, self.dim_t)  # [B, dim_t]
        t_emb = self.time_embed(t_emb)  # [B, dim_t]

        # 2. 条件输入（如果有 y）
        if self.is_y_cond and y is not None:
            if self.num_classes > 0:  # 分类任务
                y = y.squeeze()  # [B,]
            else:  # 回归任务
                y = y.view(-1, 1).float()  # [B, 1]
            t_emb += self.label_emb(y)  # [B, dim_t]

        # 3. 数据投影 + 时间编码
        x_proj = self.proj(x) + t_emb  # 加入时间信息

        # 4. MLP 预测
        return self.mlp(x_proj)  # [B, d_in]


class ResNetDiffusion(nn.Module):
    def __init__(
            self,
            d_in,  # 输入维度
            num_classes=0,  # 类别数（0 表示回归任务）
            dim_t=256,  # 时间编码维度
            n_blocks=3,  # 残差块数量
            d_main=128,  # 残差块的主维度
            d_hidden=256,  # 残差块隐藏层维度
            dropout=0.1,  # Dropout 概率
    ):
        super().__init__()
        self.dim_t = dim_t
        self.num_classes = num_classes

        # === 1. 时间编码模块 ===
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

        # === 2. 条件输入 y 的嵌入（可选）===
        if num_classes > 0:
            self.label_emb = nn.Embedding(num_classes, dim_t)  # 分类任务

        # === 3. 初始线性层 ===
        self.initial_linear = nn.Linear(d_in, d_main)

        # === 4. 残差块 ===
        self.blocks = nn.ModuleList([
            self._make_res_block(d_main, d_hidden, dropout)
            for _ in range(n_blocks)
        ])

        # === 5. 输出层 ===
        self.final_norm = nn.BatchNorm1d(d_main)
        self.final_activation = nn.SiLU()
        self.final_linear = nn.Linear(d_main, d_in)

    def _make_res_block(self, d_main, d_hidden, dropout):
        """构建一个残差块"""
        return nn.Sequential(
            nn.BatchNorm1d(d_main),
            nn.Linear(d_main, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_main),
            nn.Dropout(dropout),
        )

    def forward(self, x, timesteps, y=None):
        """
        Args:
            x: [B, d_in] 输入数据
            timesteps: [B,] 时间步
            y: [B,] 条件输入（可选）
        Returns:
            [B, d_in] 预测值
        """
        # 1. 时间编码
        t_emb = timestep_embedding(timesteps, self.dim_t)  # [B, dim_t]
        t_emb = self.time_embed(t_emb)  # [B, dim_t]

        # 2. 条件输入（如果有 y）
        if y is not None and self.num_classes > 0:
            y_emb = self.label_emb(y.squeeze())  # [B, dim_t]
            t_emb += y_emb  # 时间编码 + 条件编码

        # 3. 初始线性层
        x = self.initial_linear(x)  # [B, d_main]

        # 4. 残差块（加入时间编码）
        for block in self.blocks:
            residual = x
            x = block(x) + t_emb  # 残差连接 + 时间编码
            x = x + residual  # 跳跃连接

        # 5. 输出层
        x = self.final_norm(x)
        x = self.final_activation(x)
        return self.final_linear(x)  # [B, d_in]
