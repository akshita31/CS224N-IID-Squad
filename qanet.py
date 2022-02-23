import torch
import torch.nn as nn
from torch.nn import functional as F

class QANetConfig:
    """ base QANet config """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, block_size, **kwargs):
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)