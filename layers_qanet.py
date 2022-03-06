from layers import HighwayEncoder
from layers_char_embed import _CharEmbedding
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from util import masked_softmax

class Initialized_Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, padding=0, groups=1,
                 relu=False, bias=False):
        super(Initialized_Conv1d, self).__init__()
        self.out = nn.Conv1d(
            in_channels, out_channels,
            kernel_size, stride=stride,
            padding=padding, groups=groups, bias=bias)
        if relu is True:
            self.relu = True
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            self.relu = False
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        if self.relu is True:
            return F.relu(self.out(x))
        else:
            return self.out(x)

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
    def __init__(self, d_model, num_filters, kernel_size, num_conv_layers, num_heads, drop_prob):
        super(Encoder, self).__init__()

        self.num_filters = num_filters
        #self.positional_encoder = PositionalEncoder(d_model)
        self.drop_prob = drop_prob
        self.num_conv_layers = num_conv_layers

        #depthwise separable conv
        self.conv_layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_conv_layers)])
        self.conv_layers = nn.ModuleList([DepthwiseSeparableConv(d_model, num_filters, kernel_size) for _ in range(num_conv_layers)])

        #self attention
        self.att_layer_norm = nn.LayerNorm(d_model)
        self.att = SelfAttention(d_model, num_heads, attn_pdrop= self.drop_prob, resid_pdrop= self.drop_prob)

        #feed-forward-layers
        self.ffn_layer_norm = nn.LayerNorm(num_filters)
        #self.fc = Initialized_Conv1d(d_model, d_model, relu=True, bias=True)
        self.fc = nn.Linear(num_filters, num_filters, bias = True)


    def forward(self, x):
        # print('Input to encoder shape is', x.shape)
        (batch_size, seq_len, d_model) = x.shape

        #print("Input embedding is", x[0][5][:10])
        # out = self.positional_encoder(x)
        # out = PosEncoder(x)
        out = x
        # print('Positional Encoding shape is', out.shape)
        assert (out.size() == (batch_size, seq_len, d_model))
        #print("Positional embedding is", out[0][5][:10])
        #out = out.add(x)
        #print("Out embedding is", out[0][5][:10])
        for i, conv_layer in enumerate(self.conv_layers):
            res = out
            out = self.conv_layer_norms[i](out)
            out = torch.transpose(out, 1, 2)
            out = conv_layer(out)
            out = torch.transpose(out, 1, 2)
            out = F.relu(out)
            out = out + res
            if (i + 1) % 2 == 0:
                p_drop = self.drop_prob * (i + 1) / self.num_conv_layers
                out = F.dropout(out, p=p_drop, training=self.training)

        res = out

        # print("Output size after conv layers in encoder",out.size())
        assert (out.size() == (batch_size, seq_len, self.num_filters))
        
        out = self.att_layer_norm(out)
        out = self.att(out)
        out = out + res
        out = F.dropout(out, p=self.drop_prob, training=self.training)
        res = out

        out = self.ffn_layer_norm(out)
        out = self.fc(out)
        out = F.relu(out)
        out = out + res
        out = F.dropout(out, p=self.drop_prob, training=self.training)

        # print("Output size after fully connected layer",out.size())
        assert (out.shape == (batch_size, seq_len, self.num_filters))
        return out

class SelfAttention(nn.Module):
    def __init__(self, n_embed, n_head, attn_pdrop, resid_pdrop):
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
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels,
                                            padding=kernel_size // 2, bias=bias)
        self.pointwise_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=bias)
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.depthwise_conv.bias, 0.0)
        nn.init.kaiming_normal_(self.pointwise_conv.weight)
        nn.init.constant_(self.pointwise_conv.bias, 0.0)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


def PosEncoder(x, min_timescale=1.0, max_timescale=1.0e4):
    # x = x.transpose(1, 2)
    length = x.size()[1]
    channels = x.size()[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    return (x + signal.to(x.get_device()))


def get_timing_signal(length, channels,
                      min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim = 1)
    m = nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal
      
class QANetEmbedding(nn.Module):
    """Combines the Word and Character embedding and then applies a transformation and highway network.
    Output of this layer will be (batch_size, seq_len, hidden_size)
    """

    def __init__(self, word_vectors, char_vectors, drop_prob, d_model):
        super(QANetEmbedding, self).__init__()
        self.drop_prob = drop_prob
        self.word_embed_size = word_vectors.size(1)
        self.batch_size = word_vectors.size(0)
        self.d_model = d_model

        self.word_embed = nn.Embedding.from_pretrained(word_vectors)   
        self.char_embed = _CharEmbedding(char_vectors=char_vectors, drop_prob=drop_prob, num_filters = 100)
        self.char_embed_dim = self.char_embed.GetCharEmbedDim()
        self.conv1d = Initialized_Conv1d(self.word_embed_size + self.char_embed_dim, self.d_model, bias=False)
        self.hwy = HighwayEncoder(2, self.d_model)

    def forward(self, word_idxs, char_idxs):
        word_emb = self.word_embed(word_idxs)
        char_emb = self.char_embed(char_idxs)

        (batch_size, seq_len, _) = word_emb.shape
        assert(char_emb.shape == (batch_size, seq_len, self.char_embed_dim))
        
        word_emb = F.dropout(word_emb, self.drop_prob, self.training)
        
        emb = torch.cat((word_emb, char_emb), dim = 2)
        emb = self.conv1d(emb.transpose(1,2)).transpose(1,2)
        emb = self.hwy(emb)

        assert(emb.shape == (batch_size, seq_len, self.GetOutputEmbeddingDim()))
        return emb
    
    def GetOutputEmbeddingDim(self):
        return self.d_model
