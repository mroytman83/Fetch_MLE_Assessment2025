import torch
import torch.nn as nn


def pytorch_cos_sim(a, b):
    #typecast
    a = torch.tensor(a, dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32)

    #normalize
    a_norm = nn.functional.normalize(a, p=2, dim=1)
    b_norm = nn.functional.normalize(b, p=2, dim=1)

    return torch.mm(a_norm, b_norm.T)