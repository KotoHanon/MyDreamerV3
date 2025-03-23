import torch

def build_bins(v_min=-20, v_max=20, num_bins=256):
    bins = torch.linspace(v_min, v_max, steps=num_bins)
    return bins

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1)

def symexp(x):
    return torch.sign(x) * torch.exp(torch.abs(x) - 1)

def twohot_encode(values: torch.Tensor, bins: torch.Tensor):
    """
    values: [N, 1]
    bins: [num_bins]
    """
    bins_expanded = bins.unsqueeze(0).expand(values.size(0), -1)  # [N, num_bins]
    delta = bins[1] - bins[0]
    
    values = values.squeeze(-1) if values.dim() > 1 else values
    
    idx0 = torch.bucketize(values, bins) - 1  # [N]
    idx1 = idx0 + 1
    
    left = bins_expanded.gather(1, idx0.unsqueeze(-1)).squeeze(-1)  # [N]
    right = bins_expanded.gather(1, idx1.unsqueeze(-1)).squeeze(-1)  # [N]
    
    dist_right = (values - left).abs()
    dist_range = (right - left).abs() + 1e-8

    alpha = dist_right / dist_range
    alpha = torch.clamp(alpha, 0.0, 1.0)

    one_hot = torch.zeros(values.shape[0], bins.shape[0], device=values.device)
    one_hot.scatter_(1, idx0.unsqueeze(1), (1.0 - alpha).unsqueeze(1))
    one_hot.scatter_add_(1, idx1.unsqueeze(1), alpha.unsqueeze(1))

    return one_hot
