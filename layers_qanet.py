from layers import HighwayEncoder
from layers_char_embed import _CharEmbedding
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from util import masked_softmax

class QANetOutput(nn.Module):
    """Output layer used by QANet for question answering.

    As mentioned in the paper, output size of the encoding layers is (hidden_size = 128)
    They are basically the query informed context words representations
    
    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, d_model, drop_prob):
        super(QANetOutput, self).__init__()
        self.start_linear = nn.Linear(2*d_model, 1, bias = False)
        self.end_linear = nn.Linear(2*d_model,1 , bias = False)

    def forward(self, m0, m1, m2, mask):
        
        (batch_size, seq_len, d_model) = m0.shape

        # (batch_size, seq_len, hidden_size)
        start_enc = torch.cat((m0, m1), dim =2)
        end_enc = torch.cat((m0, m2), dim = 2)

        assert(start_enc.shape == (batch_size, seq_len, 2*d_model))
        assert(end_enc.shape == (batch_size, seq_len, 2*d_model))

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
    def __init__(self, d_model, num_filters, kernel_size, num_conv_layers, num_heads, drop_prob=0.2):
        super(Encoder, self).__init__()

        self.num_filters = num_filters
        self.positional_encoder = PositionalEncoder(d_model)

        #depthwise separable conv
        self.conv_layers = nn.ModuleList([])
        self.conv_layer_norms = nn.ModuleList([])

        for i in range(num_conv_layers):
            self.conv_layer_norms.append(nn.LayerNorm(d_model))
            self.conv_layers.append(DepthwiseSeparableConv(d_model, num_filters, kernel_size))

        #self attention
        self.att_layer_norm = nn.LayerNorm(num_filters)
        self.att = SelfAttention(num_filters, num_heads)

        #feed-forward-layers
        self.ffn_layer_norm = nn.LayerNorm(num_filters)

        # self.ffn_1 = nn.Conv1d(num_filters, num_filters, kernel_size=1)
        # nn.init.xavier_uniform_(self.ffn_1.weight)

        # self.ffn_2 = nn.Conv1d(num_filters, num_filters, kernel_size=1)
        # nn.init.xavier_uniform_(self.ffn_2.weight)

        self.fc = nn.Linear(num_filters, num_filters, bias = True)


    def forward(self, x):
        #TODO: implement residual block

        # print('Input to encoder shape is', x.shape)
        (batch_size, seq_len, d_model) = x.shape

        out = self.positional_encoder(x)
        # print('Positional Encoding shape is', out.shape)
        assert (out.size() == (batch_size, seq_len, d_model))

        out = out.add(x)

        for i, conv_layer in enumerate(self.conv_layers):
            out = self.conv_layer_norms[i](out)
            out = torch.transpose(out, 1, 2)
            out = conv_layer(out)
            out = torch.transpose(out, 1, 2)

        # print("Output size after conv layers in encoder",out.size())
        assert (out.size() == (batch_size, seq_len, self.num_filters))
        out = self.att_layer_norm(out)
        out = self.att(out)
        
        out = self.ffn_layer_norm(out)
        # out = self.ffn_1(out)
        # out = self.ffn_2(out)

        out = self.fc(out)
        out = F.relu(out)

        # print("Output size after fully connected layer",out.size())
        assert (out.shape == (batch_size, seq_len, self.num_filters))
        ## to do : modify the encoder block to add resideual
        # reference - https://github.com/heliumsea/QANet-pytorch/blob/master/models.py#L204
        return out

class SelfAttention(nn.Module):
    def __init__(self, n_embed=128, n_head=8, attn_pdrop=0.1, resid_pdrop=0.1):
        super(SelfAttention, self).__init__()
        self.n_head = n_head
        self.n_embed = n_embed
        assert self.n_embed % self.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(self.n_embed, self.n_embed)
        self.query = nn.Linear(self.n_embed, self.n_embed)
        self.value = nn.Linear(self.n_embed, self.n_embed)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(self.n_embed, self.n_embed)

        # we want to create a lower matrix so that attention is applied only to the words
        # that preceed the current word. hence creating a matrix of max_seq_len
        max_seq_len = 500
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x, layer_past=None):
        B, seq_len, C = x.size() #64, 287, 128

        # print("x_size", x.size())

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, seq_len, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, seq_len, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, seq_len, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        assert(att.shape == (B, self.n_head, seq_len, seq_len))
        # print("att shape", att.shape)

        att = att.masked_fill(self.mask[:,:,:seq_len,:seq_len] == 0, -1e10) # todo: just use float('-inf') instead?
        att = F.softmax(att, dim=-1)

        #att = masked_softmax(att, mask = mask, dim = -1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, seq_len, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))

        assert (y.shape == (B, seq_len, self.n_embed))
        return y

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):

        # Can refernce this from - https://github.com/heliumsea/QANet-pytorch/blob/master/models.py#L39
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwiseConv = nn.Conv1d(in_channels, in_channels, kernel_size, padding = kernel_size//2)
        self.pointwiseConv = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding = 0)

    def forward(self, x):
        out = self.depthwiseConv(x)
        out = self.pointwiseConv(out)
        return out

class PositionalEncoder(nn.Module):
    #Reference: https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/positional_encodings.py
    def __init__(self, in_channels):
        super(PositionalEncoder, self).__init__()

        if in_channels%2 == 0:
            self.channels = in_channels
        else:
            self.channels = in_channels + 1

        self.frequency_factor = 1.0 / (10000 ** (torch.arange(0, self.channels, 2).float() / self.channels))

    def forward(self, tensor):

        batch_size, x, orig_ch = tensor.shape
        # print("positional encoding orig shape", tensor.shape)
        pos_x = torch.arange(x, device=tensor.device).type(self.frequency_factor.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.frequency_factor)
        # print("sin_inp_x apply sincos", sin_inp_x.shape)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        #print("embx apply sincos", emb_x.shape)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        #print("emb zeros", emb.shape)
        emb[:, : self.channels] = emb_x
        #print("output", emb[None, :, :orig_ch].repeat(batch_size, 1, 1).shape)
        return emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
      
class QANetEmbedding(nn.Module):
    """Combines the Word and Character embedding and then applies a transformation and highway network.
    Output of this layer will be (batch_size, seq_len, hidden_size)
    """

    def __init__(self, word_vectors, char_vectors, drop_prob, num_filters):
        super(QANetEmbedding, self).__init__()
        self.drop_prob = drop_prob
        self.word_embed_size = word_vectors.size(1)
        self.batch_size = word_vectors.size(0)

        self.word_embed = nn.Embedding.from_pretrained(word_vectors)   
        self.char_embed = _CharEmbedding(char_vectors=char_vectors, drop_prob=drop_prob, num_filters = num_filters)
        self.char_embed_dim = self.char_embed.GetCharEmbedDim()

        self.hwy = HighwayEncoder(2, self.char_embed_dim + self.word_embed_size)
        #self.hwy = HighwayEncoder(2, (self.batch_size, self.word_embed_size, self.num_filters + self.word_embed_size))

    def forward(self, word_idxs, char_idxs):
        word_emb = self.word_embed(word_idxs)
        char_emb = self.char_embed(char_idxs)

        (batch_size, seq_len, _) = word_emb.shape
        assert(char_emb.shape == (batch_size, seq_len, self.char_embed_dim))
        
        word_emb = F.dropout(word_emb, self.drop_prob, self.training)
        
        emb = torch.cat((word_emb, char_emb), dim = 2)
        emb = self.hwy(emb)

        assert(emb.shape == (batch_size, seq_len, self.GetOutputEmbeddingDim()))
        return emb
    
    def GetOutputEmbeddingDim(self):
        return self.word_embed_size + self.char_embed_dim
