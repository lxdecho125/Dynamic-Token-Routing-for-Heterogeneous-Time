import torch
import torch.nn as nn
import torch.nn.functional as F
from models.TimePerceiver import Model as TimePerceiverModel


class MultiScaleInputEnhancer(nn.Module):
    """
    在原始时序输入上做多尺度因果卷积增强
    输入/输出形状完全相同: (B, T, C)
    """

    def __init__(self, enc_in, patch_sizes, dropout=0.1):
        super().__init__()
        self.patch_sizes = patch_sizes

        # 每个尺度：因果深度可分离卷积
        self.convs = nn.ModuleList()
        for p in self.patch_sizes:
            self.convs.append(nn.Sequential(
                nn.Conv1d(
                    in_channels  = enc_in,
                    out_channels = enc_in,
                    kernel_size  = p,
                    padding      = p - 1,    # 左侧填充，保证因果
                    groups       = enc_in,   # 深度卷积（轻量）
                    bias         = False,
                ),
                nn.Conv1d(enc_in, enc_in, kernel_size=1),
                nn.GELU(),
            ))

        # 可学习尺度融合权重
        self.scale_weights = nn.Parameter(
            torch.ones(len(patch_sizes)) / len(patch_sizes)
        )

        self.norm = nn.LayerNorm(enc_in)
        self.drop = nn.Dropout(dropout)

        # 注入门控：初始化为 -3，sigmoid(-3)≈0.05
        # 训练初期几乎等价原版，随训练自动学习注入量
        self.inject_gate = nn.Parameter(torch.tensor(-3.0))

    def forward(self, x):
        """
        x:      (B, T, C)
        return: (B, T, C)  形状完全不变
        """
        B, T, C = x.shape
        x_t = x.transpose(1, 2)          # (B, C, T)

        outputs = []
        for conv in self.convs:
            out = conv(x_t)              # (B, C, T + padding)
            out = out[:, :, :T]          # 截断到原始长度（因果）
            outputs.append(out)

        # 加权融合
        weights = torch.softmax(self.scale_weights, dim=0)
        ms_feat = sum(w * o for w, o in zip(weights, outputs))
        ms_feat = ms_feat.transpose(1, 2)    # (B, T, C)
        ms_feat = self.norm(ms_feat)
        ms_feat = self.drop(ms_feat)

        # 门控残差注入
        gate = torch.sigmoid(self.inject_gate)
        return x + gate * ms_feat


class Model(nn.Module):
    """
    MultiScaleTimePerceiver v4

    与原版 TimePerceiver 唯一区别：
      forward() 开头对输入做多尺度增强（残差门控）
      之后完全调用 base.forward()

    完全兼容：
      ✅ Standard 路径
      ✅ Generalized 路径（random_split_patches 的任意 indices）
      ✅ RevIN（在 base.forward 内部处理）
    """

    def __init__(self, configs):
        super().__init__()

        # 完整复用原版 TimePerceiver（所有参数共享）
        self.base = TimePerceiverModel(configs)

        # patch_sizes 解析
        raw = getattr(configs, 'patch_sizes', None)
        if raw is None:
            base_p = configs.patch_len
            patch_sizes = [max(2, base_p // 2), base_p * 2]
        elif isinstance(raw, str):
            patch_sizes = [int(p) for p in raw.split(',')]
        else:
            patch_sizes = list(raw)

        # 去掉与 patch_len 重复的尺度（原版已处理该尺度）
        patch_sizes = [p for p in patch_sizes if p != configs.patch_len]

        print(f"[MultiScaleTimePerceiver v4] "
              f"patch_len={configs.patch_len}, "
              f"extra_scales={patch_sizes}")

        # 多尺度输入增强器
        if len(patch_sizes) > 0:
            self.enhancer = MultiScaleInputEnhancer(
                enc_in      = configs.enc_in,
                patch_sizes = patch_sizes,
                dropout     = configs.dropout,
            )
            self.use_enhancer = True
        else:
            self.use_enhancer = False
            print("[MultiScaleTimePerceiver v4] 无额外尺度，等价原版")                                          # (B, C, actual_L, D)

    # ─────────────────────────────────────────────────────────────────────
    def forward(self, inputs, x_mark_enc, x_dec, x_mark_dec,
                indices=None, mask=None):

        # ── RevIN（复用 base 模型参数）────────────────────────────────────
        means  = inputs.mean(1, keepdim=True).detach()
        inputs = inputs - means
        stdev  = torch.sqrt(
            torch.var(inputs, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        inputs = inputs / stdev

        B = inputs.shape[0]

        # ── 获取 past indices ─────────────────────────────────────────────
        base = self.base
        if indices is not None:
            past_idx   = indices[0]
            future_idx = indices[1]
        else:
            past_idx   = list(range(self.past_patch_num))
            future_idx = list(range(
                self.past_patch_num,
                self.past_patch_num + self.pred_len // self.patch_len
            ))

        # ── 多尺度 Patch Embedding（带旁路注入）──────────────────────────
        inputs_emb  = self._patch_embed_with_ms(inputs, indices)
        in_channels = inputs_emb.shape[1]
        patch_num   = inputs_emb.shape[2]

        # ── 以下与原版 TimePerceiver.forward 完全相同 ─────────────────────
        inputs_flat = inputs_emb.view(B, in_channels * patch_num, self.d_model)

        if base.use_latent:
            latent = base.latent_array.expand(B, -1, -1)
            for _ in range(base.num_latent_blocks):
                latent      = base.latent_cross_attention(latent, inputs_flat)
                for block in base.latent_attention_blocks:
                    latent  = block(latent, latent)
                inputs_flat = base.write_cross_attention(inputs_flat, latent)

        if base.query_share:
            query = (
                base.patch_positional_embedding[:, :, future_idx, :]
                + base.channel_positional_embedding
            )
        else:
            query = base.query[:, :, future_idx, :]

        query = query.expand(B, -1, -1, -1).contiguous().reshape(
            B * in_channels, -1, self.d_model
        )
        inputs_per_ch = inputs_flat.view(
            B, in_channels, patch_num, self.d_model
        ).contiguous().reshape(B * in_channels, -1, self.d_model)

        outputs = base.query_cross_attention(query, inputs_per_ch)
        outputs = outputs.reshape(B, in_channels, -1, self.d_model)
        outputs = base.output_projection(outputs)
        outputs = outputs.view(B, in_channels, -1).permute(0, 2, 1)
        outputs = outputs[:, :self.pred_len, :]

        # ── RevIN 逆变换 ──────────────────────────────────────────────────
        outputs = outputs * stdev[:, 0, :].unsqueeze(1).expand(-1, self.pred_len, -1)
        outputs = outputs + means[:, 0, :].unsqueeze(1).expand(-1, self.pred_len, -1)

        return outputs
