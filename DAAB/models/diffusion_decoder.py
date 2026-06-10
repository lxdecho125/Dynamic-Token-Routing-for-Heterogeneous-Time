import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 时间步嵌入（扩散模型核心，将离散时间步转为连续嵌入）
class TimeStepEmbedding(nn.Module):
    def __init__(self, embed_dim, max_steps=1000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_steps = max_steps
        # 预计算频率
        half_dim = embed_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half_dim) / half_dim)
        self.register_buffer('freqs', freqs)

    def forward(self, t):
        # t: [batch_size]
        t = t.float()
        emb = t.unsqueeze(1) * self.freqs.unsqueeze(0)  # [B, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [B, embed_dim]
        if self.embed_dim % 2 == 1:
            emb = F.pad(emb, (0,1))
        return emb

# 时间序列去噪网络（核心：带条件约束的MLP+注意力，适配原模型的embed_dim）
class TS_DenoiseNet(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.embed_dim = configs.d_model  # 与原模型一致
        self.patch_len = configs.patch_len  # 与原模型一致
        self.n_heads = configs.n_heads
        self.dropout = configs.dropout

        # 时间步嵌入层
        self.t_emb = TimeStepEmbedding(self.embed_dim)
        self.t_emb_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # 条件约束交叉注意力（融合原模型的input特征，保证预测不偏离时序趋势）
        self.cond_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.n_heads,
            dropout=self.dropout,
            batch_first=True
        )
        self.cond_norm = nn.LayerNorm(self.embed_dim)

        # 去噪MLP（适配一维时间序列）
        self.denoise_mlp = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim * 4, self.embed_dim),
            nn.Dropout(self.dropout)
        )

    def forward(self, x_t, t, cond_feat):
        """
        x_t: 带噪的未来patch特征 [B*C, future_P_N, embed_dim]
        t: 扩散时间步 [B]
        cond_feat: 原模型的编码特征（条件约束） [B*C, past_P_N, embed_dim]
        """
        # 时间步嵌入
        t_emb = self.t_emb(t)  # [B, embed_dim]
        t_emb = self.t_emb_proj(t_emb)  # [B, embed_dim]
        t_emb = t_emb.unsqueeze(1).expand(-1, x_t.shape[1], -1)  # [B, future_P_N, embed_dim]
        # 适配B*C的batch维度
        t_emb = t_emb.repeat_interleave(cond_feat.shape[0] // t_emb.shape[0], dim=0)

        # 融合时间步嵌入到带噪特征
        x_t = x_t + t_emb

        # 条件约束：融合原模型的编码特征（核心，保证扩散预测贴合原时序）
        attn_out, _ = self.cond_attn(x_t, cond_feat, cond_feat)
        x_t = self.cond_norm(x_t + attn_out)

        # 去噪预测
        x_0_hat = self.denoise_mlp(x_t)
        return x_0_hat

# 核心：时间序列扩散解码器（替换原查询解码器）
class DiffusionQueryDecoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.embed_dim = configs.d_model
        self.future_patch_num = configs.pred_len // configs.patch_len
        self.patch_len = configs.patch_len
        self.device = torch.device('cuda:{}'.format(configs.gpu)) if configs.use_gpu else torch.device('cpu')

        # 扩散超参（可通过args配置，默认轻量级：50步DDIM，降低显存/速度损耗）
        self.num_diffusion_steps = configs.get('diffusion_steps', 50)
        self.beta_start = configs.get('beta_start', 0.0001)
        self.beta_end = configs.get('beta_end', 0.02)
        self.ddim_eta = configs.get('ddim_eta', 0.0)  # DDIM采样，eta=0为确定性采样

        # 初始化扩散beta/alpha参数
        self._init_diffusion_params()

        # 去噪网络（核心）
        self.denoise_net = TS_DenoiseNet(configs)

        # 原模型的输出投影层（复用，保证输出维度匹配patch_len）
        self.output_projection = nn.Linear(self.embed_dim, configs.patch_len)

    def _init_diffusion_params(self):
        """预计算扩散过程的beta/alpha/累计乘积，注册为buffer（不参与训练）"""
        betas = torch.linspace(self.beta_start, self.beta_end, self.num_diffusion_steps, device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), alphas_cumprod[:-1]])
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # 预计算扩散过程的其他参数
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1. / alphas))
        self.register_buffer('recip_sqrt_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod_prev', torch.sqrt(1. - alphas_cumprod_prev))

    def _add_noise(self, x_0, t, noise=None):
        """给干净特征x_0添加t步噪声，得到x_t"""
        if noise is None:
            noise = torch.randn_like(x_0, device=self.device)
        x_t = self.sqrt_alphas_cumprod[t, None, None] * x_0 + self.sqrt_one_minus_alphas_cumprod[t, None, None] * noise
        return x_t, noise

    @torch.no_grad()
    def ddim_sample(self, cond_feat, batch_size):
        """
        DDIM采样：从纯噪声逐步去噪，生成未来patch的干净特征
        cond_feat: 原模型的编码特征 [B*C, past_P_N, embed_dim]
        return: 去噪后的未来patch特征 [B*C, future_P_N, embed_dim]
        """
        # 初始化纯噪声（匹配未来patch的维度）
        x_t = torch.randn(
            (cond_feat.shape[0], self.future_patch_num, self.embed_dim),
            device=self.device
        )
        # 逆序时间步去噪
        for t in reversed(range(0, self.num_diffusion_steps)):
            t_tensor = torch.tensor([t], device=self.device).repeat(batch_size)
            # 去噪网络预测干净特征
            x_0_hat = self.denoise_net(x_t, t_tensor, cond_feat)
            # DDIM采样更新x_t
            alpha_t = self.alphas[t]
            alpha_t_prev = self.alphas_cumprod_prev[t]
            alpha_cumprod_t = self.alphas_cumprod[t]

            # 计算DDIM的均值和方差
            pred_noise = (x_t - torch.sqrt(alpha_cumprod_t) * x_0_hat) / torch.sqrt(1. - alpha_cumprod_t)
            mean = (x_t - pred_noise * (1 - alpha_t) / self.sqrt_one_minus_alphas_cumprod[t]) / torch.sqrt(alpha_t)
            mean = mean + torch.sqrt(alpha_t_prev) * (1 - alpha_t) / self.sqrt_one_minus_alphas_cumprod[t] * pred_noise

            if t > 0:
                variance = self.ddim_eta * (1 - alpha_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_t_prev)
                noise = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(variance) * noise
            else:
                x_t = mean
        return x_t

    def forward(self, cond_feat, batch_size, is_train=True):
        """
        扩散解码器前向传播：训练时加噪去噪，推理时DDIM采样
        cond_feat: 原模型的编码特征 [B*C, past_P_N, embed_dim]
        batch_size: 原批次大小B（非B*C）
        is_train: True=训练（加噪去噪），False=推理（DDIM采样）
        """
        if is_train:
            # 训练阶段：用原查询的目标特征作为x_0，加噪后去噪（有监督训练）
            # 生成随机时间步
            t = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=self.device)
            # 生成未来patch的干净特征（原查询的target，这里用随机初始化的真实特征，训练时由原模型传入）
            x_0 = torch.randn_like(torch.zeros(cond_feat.shape[0], self.future_patch_num, self.embed_dim), device=self.device)
            # 加噪
            x_t, noise = self._add_noise(x_0, t)
            # 去噪预测
            x_0_hat = self.denoise_net(x_t, t, cond_feat)
            return x_0_hat, noise  # 返回预测值和噪声，用于计算去噪损失
        else:
            # 推理阶段：DDIM采样生成未来patch特征
            x_0_hat = self.ddim_sample(cond_feat, batch_size)
            return x_0_hat