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
    def __init__(self, input_size=500, num_filters=128, kernel_size=7, num_conv_layers=4, num_heads=8, drop_prob=0.1):
        super(Encoder, self).__init__()

        self.positional_encoder = PositionalEncoder(input_size)
        
        #depthwise separable conv
        self.conv_layers = nn.ModuleList([])
        self.conv_layer_norms = nn.ModuleList([])
        for i in range(num_conv_layers):
            if i==0:
                self.conv_layer_norms.append(nn.LayerNorm(input_size))
            else:
                self.conv_layer_norms.append(nn.LayerNorm(num_filters))

            self.conv_layers.append(DepthwiseSeparableConv(input_size, num_filters, kernel_size))

        #self attention
        self.att_layer_norm = nn.LayerNorm(num_filters)
        self.att = SelfAttention(num_filters, num_heads)

        #feed-forward-layers
        self.ffn_layer_norm = nn.LayerNorm(num_filters)

        self.ffn_1 = nn.Conv1d(num_filters, num_filters, kernel_size=1)
        nn.init.xavier_uniform_(self.ffn_1)

        self.ffn_2 = nn.Conv1d(num_filters, num_filters, kernel_size=1)
        nn.init.xavier_uniform_(self.ffn_2)


    def forward(self, x):
        #TODO: implement residual block
        out = self.positional_encoder.forward(x)
        for i, conv_layer in enumerate(self.conv_layers):
            out = self.conv_layer_norms[i](out)
            out = conv_layer(out)

        out = self.att_layer_norm(out)
        out = self.att(out)
        
        out = self.ffn_layer_norm(out)
        out = self.ffn_1(out)
        out = self.ffn_2(out)

        return out

class SelfAttention(nn.Module):
    def __init__(self, n_embed=128, n_head=8):
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

class PositionalEncoder(nn.Module):
    #Reference: https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/positional_encodings.py
    def __init__(self, in_channels):
        super.(PositionalEncoding, self).__init__()

        if in_channels%2 == 0:
            self.channels = in_channels
        else:
            self.channels = in_channels + 1

        self.frequency_factor = 1.0 / (10000 ** (torch.arange(0, self.channels, 2).float() / self.channels))

    def forward(self, x):
        pos_x = torch.arange(x)
        frequency_x = torch.einsum("i,j->ij", pos_x, self.frequency_factor)
        emb_x = torch.cat((frequency_x.sin(), frequency_x.cos()), dim=-1)

        emb = torch.zeros((x, self.channels))
        emb[:, : self.channels] = emb_x
        
        return emb[None, :, :x.shape]
