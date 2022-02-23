import torch
import torch.nn as nn
import torch.nn.functional as f

from util import masked_softmax

class QANetOutput(nn.Module):
    """Output layer used by QANet for question answering.

    As mentioned in the paper, output size of the encoding layers is (hidden_size = 128)
    They are basically the query informed context words representations
    
    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(QANetOutput, self).__init__()
        self.start_linear = nn.Linear(2*hidden_size, 1, bias = False)
        self.end_linear = nn.Linear(2*hidden_size,1 , bias = False)

    def forward(self, m0, m1, m2, mask):
        
        (batch_size, seq_len, hidden_size) = m0.shape

        # (batch_size, seq_len, hidden_size)
        start_enc = torch.cat((m0, m1), dim =2)
        end_enc = torch.cat((m0, m2), dim = 2)

        assert(start_enc.shape == (batch_size, seq_len, 2*hidden_size))
        assert(end_enc.shape == (batch_size, seq_len, 2*hidden_size))

        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.start_linear(start_enc)
        logits_2 = self.end_linear(end_enc)

        assert(logits_1.shape == (batch_size, seq_len, 1))
        assert(logits_2.shape == (batch_size, seq_len, 1))

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(dim=2), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(dim=2), mask, log_softmax=True)

        return log_p1, log_p2

class Encoder(nn.Module):
    def __init__(self, input_size):
        super(Encoder, self).__init__()
        #depthwise separable conv
        #self attention
        self.FFN1 = nn.Conv1d(input_size, input_size, 1)
        nn.init.xavier_uniform_(self.FFN1)

        self.FFN2 = nn.Conv1d(input_size, input_size, 1)
        nn.init.xavier_uniform_(self.FFN2)


    def forward(self):
        pass

class SelfAttention(nn.Module):
    def __init__(self, n_head=8, n_embed):
        super(SelfAttention, self).__init__()
        self.n_head = n_head
        self.n_embed = n_embed
        assert self.n_embd % self.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(self.n_embed, self.n_embed)
        self.query = nn.Linear(self.n_embed, self.n_embed)
        self.value = nn.Linear(self.n_embed, self.n_embed)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(self.n_embed, self.n_embed)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e10) # todo: just use float('-inf') instead?
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwiseConv = nn.Conv1d(in_channels, in_channels, kernel_size, groups=in_channels)
        self.pointwiseConv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwiseConv(x)
        out = self.pointwiseConv(out)
        return out

# class FFN(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size):
#         super(FFN, self).__init__()
#         self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size)
#         nn.init.xavier_uniform_(self.)

#     def forward(self, x):
#         return self.conv1d(x)  