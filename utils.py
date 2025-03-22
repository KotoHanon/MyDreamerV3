import torch

def build_bins(v_min=-20, v_max=20, num_bins=256):
    bins = torch.linspace(v_min, v_max, steps=num_bins)
    return bins

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1)

def symexp(x):
    return torch.sign(x) * torch.exp(torch.abs(x) - 1)
