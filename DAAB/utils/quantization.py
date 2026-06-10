import numpy as np
import torch

def compute_bins(data, num_classes):
    """根据数据计算等频分箱边界"""
    data_flat = data.flatten()
    percentiles = np.linspace(0, 100, num_classes + 1)[1:-1]  # 内部边界
    bins = np.percentile(data_flat, percentiles)
    bins = np.concatenate([[-np.inf], bins, [np.inf]])
    return bins

def values_to_classes_and_residuals(values, bins):
    """将连续值转换为类别索引和区间内残差（相对于左边界）"""
    # values: (batch, seq_len, channels) or any shape
    device = values.device
    bins = bins.to(device)
    # 找到每个值所在的区间索引
    indices = torch.bucketize(values, bins) - 1  # bucketize返回1-based索引
    indices = torch.clamp(indices, 0, len(bins)-2)  # 防止越界
    left = bins[indices]
    residuals = values - left
    return indices, residuals