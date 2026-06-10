import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence
import math
# 补充NumPy导入（放在import torch之后即可）
import numpy as np
from numpy import fft
import torch
import torch.nn as nn
import torch
import torch.nn as nn
# 如果顶部没有引入 torch.fft，请确保引入 (PyTorch 1.8+ 默认支持)

class ComplexityEvaluator(nn.Module):
    """
    升级版：时-频-通道 三域联合感知复杂度评估器 (Time-Frequency-Channel Dual-Awareness)
    """
    def __init__(self, seq_len, in_channels, max_latents, latent_dim):
        super(ComplexityEvaluator, self).__init__()
        self.max_latents = max_latents
        self.latent_dim = latent_dim
        
        # 1. 时间域特征提取器 (评估时域局部波动)
        self.time_encoder = nn.Sequential(
            nn.Linear(seq_len, 32),
            nn.GELU(),
            nn.Linear(32, 16)
        )
        
        # 2. 通道域特征提取器 (评估跨通道全局活跃度)
        self.channel_encoder = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.GELU(),
            nn.Linear(32, 16)
        )
        
        # ================== 新增：频率域特征提取器 ==================
        # 实数 FFT (rfft) 的输出长度固定为 seq_len // 2 + 1
        freq_len = seq_len // 2 + 1  
        self.freq_encoder = nn.Sequential(
            nn.Linear(freq_len, 32),
            nn.GELU(),
            nn.Linear(32, 16)
        )
        # ==========================================================
        
        # 4. 门控生成器：融合三大域特征 (16 + 16 + 16 = 48)
        self.gate_net = nn.Sequential(
            nn.Linear(48, 64),
            nn.GELU(),
            nn.Linear(64, max_latents + latent_dim),
            nn.Sigmoid() 
        )
        
        # 修复冷启动：强行让初始门控值接近 1.0 (防止专家早夭)
        nn.init.constant_(self.gate_net[2].weight, 0.0)
        nn.init.constant_(self.gate_net[2].bias, 3.0)

    def forward(self, x):
        # x: [Batch, Seq_Len, Channels] 
        
        # --- 域 1：时间域 (Time Domain) ---
        x_time_view = torch.sqrt(torch.mean(x ** 2, dim=2) + 1e-5) # RMS [B, S]
        time_feat = self.time_encoder(x_time_view) # -> [B, 16]
        
        # --- 域 2：通道域 (Channel Domain) ---
        x_channel_view = x.std(dim=1) # STD [B, C]
        x_channel_view = torch.nan_to_num(x_channel_view, 0.0) 
        channel_feat = self.channel_encoder(x_channel_view) # -> [B, 16]
        
        # --- 域 3：频率域 (Frequency Domain - 新增) ---
        x_mean_c = x.mean(dim=2) # 消除通道干扰，保留主干时序 [B, S]
        x_freq = torch.fft.rfft(x_mean_c, dim=1) # 实数快速傅里叶变换 -> [B, S//2+1] (复数)
        x_freq_amp = torch.log(torch.abs(x_freq) + 1.0) # 取振幅 (Magnitude)，反映各频率周期的能量大小
        freq_feat = self.freq_encoder(x_freq_amp) # -> [B, 16]
        
        # --- 三域特征融合与门控生成 ---
        # Concat 拼接: 16 (Time) + 16 (Channel) + 16 (Freq) = 48 维
        joint_feat = torch.cat([time_feat, channel_feat, freq_feat], dim=1) # -> [B, 48]
        gates = self.gate_net(joint_feat)
        
        token_gate = gates[:, :self.max_latents] # [B, M] 控制令牌数量
        dim_gate = gates[:, self.max_latents:]   # [B, D] 控制维度权重
        
        return token_gate, dim_gate

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.patch_size = configs.patch_len
        self.past_patch_num = configs.seq_len // configs.patch_len
        self.future_patch_num = configs.pred_len // configs.patch_len
        self.embed_dim = configs.d_model
        self.pred_len = configs.pred_len
        self.query_share = configs.query_share

        # Latent
        self.use_latent = configs.use_latent
        self.num_latents = configs.num_latents
        self.latent_dim = configs.latent_dim
        self.num_latent_blocks = configs.num_latent_blocks
        self.latent_array = nn.Parameter(torch.randn(1, self.num_latents, self.latent_dim))
        if self.use_latent:
            self.complexity_evaluator = ComplexityEvaluator(
                seq_len=configs.seq_len, 
                in_channels=configs.enc_in, 
                max_latents=self.num_latents, 
                latent_dim=self.latent_dim
            )
            self.gate_norm = nn.LayerNorm(self.latent_dim)

        # Positional embedding(time, channel directions)
        self.patch_positional_embedding = nn.Parameter(torch.randn(1, 1, self.past_patch_num + self.future_patch_num, self.embed_dim))
        self.channel_positional_embedding = nn.Parameter(torch.randn(1, configs.enc_in, 1, self.embed_dim))

        self.patch_embedding = nn.Linear(configs.patch_len, self.embed_dim)

        if not configs.query_share:
            self.query = nn.Parameter(torch.randn(1, configs.enc_in, self.past_patch_num + self.future_patch_num, self.embed_dim))

        self.latent_cross_attention = AttentionBlock(configs.n_heads, self.latent_dim, self.embed_dim, configs.latent_d_ff, configs.dropout)
        self.latent_attention_blocks = nn.ModuleList([
            AttentionBlock(configs.n_heads, self.latent_dim, self.latent_dim, configs.latent_d_ff, configs.dropout)
            for _ in range(3)
        ])
        self.write_cross_attention = AttentionBlock(configs.n_heads, self.embed_dim, self.latent_dim, configs.d_ff, configs.dropout)
        self.query_cross_attention = AttentionBlock(configs.n_heads, self.embed_dim, self.embed_dim, configs.d_ff, configs.dropout)

        self.output_projection = nn.Linear(self.embed_dim, configs.patch_len)

    def forward(self, inputs, x_mark_enc, x_dec, x_mark_dec, indices=None, mask=None):
        # RevIN
        means = inputs.mean(1, keepdim=True).detach()
        inputs = inputs - means
        stdev = torch.sqrt(
            torch.var(inputs, dim=1, keepdim=True, unbiased=False) + 1e-5)
        inputs /= stdev
        inputs_for_eval = inputs.clone()

        # Patching (B, S, C) -> (B, C, P_N, D)
        inputs = inputs.transpose(1, 2)
        inputs = inputs.unfold(2, self.patch_size, self.patch_size)
        batch_size, in_channels, patch_num, patch_size = inputs.size()
        inputs = self.patch_embedding(inputs)

        # Add TPE, CPE to input
        if indices:
            inputs = inputs + self.patch_positional_embedding[:, :, indices[0], :]
        else:
            inputs = inputs + self.patch_positional_embedding[:, :, :self.past_patch_num, :]
        inputs = inputs + self.channel_positional_embedding
        inputs = inputs.view(batch_size, in_channels * patch_num, self.embed_dim) # (B, C * P_N, D)

        # Input, latent cross attention
        if self.use_latent:
            token_gate, dim_gate = self.complexity_evaluator(inputs_for_eval)
            self.current_token_gate = token_gate
            latent = self.latent_array.expand(batch_size, -1, -1)
            if np.random.rand() < 0.01: 
                print(f"当前 Token 平均门控值 (活跃度): {token_gate.mean().item():.4f}")
            
            # 创新点③：动态维度分配
            latent = latent * dim_gate.unsqueeze(1)
            latent = latent * token_gate.unsqueeze(-1)
            #inputs = self.write_cross_attention(inputs, latent)
            latent = self.gate_norm(latent)

            
            # ====== 修改：暂时关闭推理时的硬剪枝 ======
            # if not self.training: 
            #     score = token_gate.mean(dim=0)
            #     active_idx = torch.where(score > 0.1)[0]
            #     if len(active_idx) == 0: active_idx = torch.tensor([0], device=latent.device)
            #     latent = latent[:, active_idx, :] 
            # ==========================================
            
            # 执行交互
            for _ in range(self.num_latent_blocks):
                latent = self.latent_cross_attention(latent, inputs)
                for block in self.latent_attention_blocks:
                    latent = block(latent, latent)
                inputs = self.write_cross_attention(inputs, latent)


        # Configure the query
        if self.query_share:
            if indices:
                query = self.patch_positional_embedding[:, :, indices[1], :] + self.channel_positional_embedding
            else:
                query = self.patch_positional_embedding[:, :, self.past_patch_num:, :] + self.channel_positional_embedding
        else:
            query = self.query[:, :, indices[1], :]

        query = query.expand(batch_size, -1, -1, -1).contiguous().reshape(batch_size * in_channels, -1, self.embed_dim) # (B, C, Future P_N, E_D)
        inputs = inputs.view(batch_size, in_channels, patch_num, self.embed_dim).contiguous().reshape(batch_size * in_channels, -1, self.embed_dim)

        outputs = self.query_cross_attention(query, inputs) # (B * C, Future P_N, E_D)
        outputs = outputs.reshape(batch_size, in_channels, -1, self.embed_dim)

        outputs = self.output_projection(outputs)

        outputs = outputs.view(batch_size, in_channels, -1).contiguous().permute(0, 2, 1) # (B, S, C)

        outputs = outputs * \
            (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        outputs = outputs + \
            (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return outputs

class CrossAttention(nn.Module):
    def __init__(self, num_heads, query_dim, key_value_dim, dropout_rate):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.key_value_dim = key_value_dim
        self.query_dim = query_dim
        self.dropout = nn.Dropout(dropout_rate)

        if key_value_dim % num_heads != 0:
            raise ValueError("The hidden size must be a multiple of the num_heads size.")

        self.head_size = query_dim // num_heads

        self.query = nn.Linear(query_dim, query_dim)
        self.key = nn.Linear(key_value_dim, query_dim)
        self.value = nn.Linear(key_value_dim, query_dim)
        self.out = nn.Sequential(nn.Linear(query_dim, query_dim), self.dropout)

    def forward(self, query_input, key_value_input):
        batch_size = query_input.shape[0]
        query_len = query_input.shape[1]
        key_value_len = key_value_input.shape[1]

        query = self.query(query_input)
        key = self.key(key_value_input)
        value = self.value(key_value_input)

        query = query.view(batch_size, query_len, self.num_heads, self.head_size).transpose(1, 2)  # (batch_size, num_heads, query_len, head_size)
        key = key.view(batch_size, key_value_len, self.num_heads, self.head_size).transpose(1, 2)  # (batch_size, num_heads, key_value_len, head_size)
        value = value.view(batch_size, key_value_len, self.num_heads, self.head_size).transpose(1, 2)  # (batch_size, num_heads, key_value_len, head_size)

        score_matrix = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32))
        attention_matrix = torch.softmax(score_matrix, dim=-1)
        attention_matrix = self.dropout(attention_matrix)

        result_matrix = torch.matmul(attention_matrix, value)

        result_matrix = result_matrix.transpose(1, 2).contiguous().view(batch_size, query_len, self.query_dim)

        out = self.out(result_matrix)

        return out

class AttentionBlock(nn.Module):
    def __init__(self, num_heads, query_dim, key_value_dim, mlp_dim, dropout_rate):
        super(AttentionBlock, self).__init__()
        self.query_norm = nn.LayerNorm(query_dim)
        self.key_value_norm = nn.LayerNorm(key_value_dim)

        self.attention = CrossAttention(num_heads, query_dim, key_value_dim, dropout_rate)
        self.layer_norm2 = nn.LayerNorm(query_dim)

        self.dropout = nn.Dropout(dropout_rate)

        self.mlp = nn.Sequential(
            nn.Linear(query_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, query_dim),
        )

    def forward(self, query_input, key_value_input):
        query_input = query_input + self.dropout(self.attention(self.query_norm(query_input), self.key_value_norm(key_value_input)))
        query_input = query_input + self.dropout(self.mlp(self.layer_norm2(query_input)))

        return query_input