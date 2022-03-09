import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import args
from util import masked_softmax

train_args = args.get_train_args()
Dword = train_args.glove_dim
Dchar = train_args.char_dim
D = train_args.d_model
dropout = train_args.qanet_dropout
dropout_char = train_args.qanet_char_dropout


def mask_logits(inputs, mask):
    mask = mask.type(torch.float32)
    return inputs + (-1e30) * (1 - mask)


class Initialized_Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, relu=False, 
                 stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.out = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, groups=groups, bias=bias)
        if relu is True:
            self.relu = True
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            self.relu = False
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        if self.relu == True:
            return F.relu(self.out(x))
        else:
            return self.out(x)

#A positional encoding is added to the input at the beginning of each encoder layer consisting of sin and cos functions at varying wavelengths,
# as defined in (Vaswani et al., 2017a). Each sub-layer after the positional encoding (one of convolution, self-attention, or feed-forward-net)
# inside the encoder structure is wrapped inside a residual block. (Page 3 of the paper).
def PosEncoder(x, min_timescale=1.0, max_timescale=1.0e4):
    x = x.transpose(1, 2)
    length, channels = x.shape[1], x.shape[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    if torch.cuda.is_available():
        signal = signal.cuda()
    return (x + signal).transpose(1,2)


def get_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales)-1)
    )
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim = 1)
    m = nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(
            in_channels=in_ch, out_channels=in_ch, kernel_size=k, 
            groups=in_ch, padding=k // 2, bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, 
                                        kernel_size=1, padding=0, bias=bias)
    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))

class HighwayEncoder(nn.Module):
    def __init__(self, layer_num: int, size=D):
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([
            Initialized_Conv1d(size, size, relu=False, bias=True) 
            for _ in range(self.n)
        ])
        self.gate = nn.ModuleList([
            Initialized_Conv1d(size, size, bias=True) for _ in range(self.n)])

    def forward(self, x):
        #x: shape [batch_size, hidden_size, length]
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = self.linear[i](x)
            nonlinear = F.dropout(nonlinear, p=dropout, training=self.training)
            x = gate * nonlinear + (1 - gate) * x
            #x = F.relu(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, n_head=8):
        super().__init__()
        # self.n_head = n_head
        # self.n_embed = n_embed
        self.n_head = n_head
        self.mem_conv = Initialized_Conv1d(
            D, D * 2, kernel_size=1, relu=False, bias=False)
        self.query_conv = Initialized_Conv1d(
            D, D, kernel_size=1, relu=False, bias=False)
        # key, query, value projections for all heads
        # self.key = nn.Linear(self.n_embed, self.n_embed)
        # self.query = nn.Linear(self.n_embed, self.n_embed)
        # self.value = nn.Linear(self.n_embed, self.n_embed)
        bias = torch.empty(1)
        #  regularization
        #  self.attn_drop = nn.Dropout(attn_pdrop)
        #  self.resid_drop = nn.Dropout(resid_pdrop)
        #  output projection
        #  self.proj = nn.Linear(self.n_embed, self.n_embed)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)
        #  we want to create a lower matrix so that attention is applied only to the words
        #  that preceed the current word. hence creating a matrix of max_seq_len
        #  max_seq_len = 500
        #  causal mask to ensure that attention is only applied to the left in the input sequence
        #  self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len))

    def forward(self, queries, mask):
        memory = queries
        # B, seq_len, C = x.size() #64, 287, 128
        memory = self.mem_conv(memory)
        query = self.query_conv(queries)
        memory = memory.transpose(1, 2)
        query = query.transpose(1, 2)
        # # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # k = self.key(x).view(B, seq_len, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # q = self.query(x).view(B, seq_len, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # v = self.value(x).view(B, seq_len, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        Q = self.split_last_dim(query, self.n_head)
        K, V = [
            self.split_last_dim(tensor, self.n_head) 
            for tensor in torch.split(memory, D, dim=2)
        ]
        # att = att.masked_fill(self.mask[:,:,:seq_len,:seq_len] == 0, -1e10) # todo: just use float('-inf') instead?
        # att = F.softmax(att, dim=-1)
        # att = self.attn_drop(att)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # y = y.transpose(1, 2).contiguous().view(B, seq_len, C) # re-assemble all head outputs side by side
        # y = self.resid_drop(self.proj(y))
        # return y


        key_depth_per_head = D // self.n_head
        Q *= key_depth_per_head**-0.5
        x = self.dot_product_attention(Q, K, V, mask = mask)
        return self.combine_last_two_dim(x.permute(0,2,1,3)).transpose(1, 2)


    def dot_product_attention(self, q, k ,v, bias = False, mask = None):
        """dot-product attention.
        """
        # # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # assert(att.shape == (B, self.n_head, seq_len, seq_len))
        logits = torch.matmul(q,k.permute(0,1,3,2))
        if bias:
            logits += self.bias
        if mask is not None:
            shapes = [x  if x != None else -1 for x in list(logits.size())]
            mask = mask.view(shapes[0], 1, 1, shapes[-1])
            logits = mask_logits(logits, mask)
        weights = F.softmax(logits, dim=-1)
        # dropping out the attention links for each of the heads
        weights = F.dropout(weights, p=dropout, training=self.training)
        return torch.matmul(weights, v)

    def split_last_dim(self, x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        """
        old_shape = list(x.size())
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.view(new_shape)
        return ret.permute(0, 2, 1, 3)

    def combine_last_two_dim(self, x):
        """Reshape x so that the last two dimension become one.
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.contiguous().view(new_shape)
        return ret


#We adopt the standard techniques to obtain the embedding of each word w by concatenating its word embedding and character embedding.
# The word embedding is fixed during training and initialized from the p1 = 300 dimensional pre-trained GloVe (Pennington et al., 2014) word vectors,
# which are fixed during training. All the out-of-vocabulary words are mapped to an <UNK> token, whose embedding is trainable with
# random initialization. The character embedding is obtained as follows: Each character is represented as a trainable vector
# of dimension p2 = 200, meaning each word can be viewed as the concatenation of the embedding vectors for each of its characters.
# The length of each word is either truncated or padded to 16. We take maximum value of each row of this matrix to get a fixed-size
# vector representation of each word. Finally, the output of a given word x from this layer is the concatenation [xw;xc] ∈ Rp1+p2,
# where xw and xc are the word embedding and the convolution output of character embedding of x respectively. Following Seo et al.
# (2016), we also adopt a two-layer highway network (Srivastava et al., 2015) on top of this representation. For simplicity,
# we also use x to denote the output of this layer.

class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(
            Dchar, D, kernel_size=(1,5), padding=0, bias=True)
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')
        self.conv1d = Initialized_Conv1d(Dword+D, D, bias=False)
        self.high = HighwayEncoder(2)

    def forward(self, ch_emb, wd_emb):
        ch_emb = ch_emb.permute(0, 3, 1, 2)
        ch_emb = F.dropout(ch_emb, p=dropout_char, training=self.training)
        ch_emb = self.conv2d(ch_emb)
        ch_emb = F.relu(ch_emb)
        ch_emb, _ = torch.max(ch_emb, dim=3)
        ch_emb = ch_emb.squeeze()

        wd_emb = F.dropout(wd_emb, p=dropout, training=self.training)
        wd_emb = wd_emb.transpose(1, 2)
        emb = torch.cat([ch_emb, wd_emb], dim=1)
        emb = self.conv1d(emb)
        emb = self.high(emb)
        return emb

class _CharEmbedding(nn.Module):
    """Character Embedding layer used by BiDAF.

    It takes in an input word (or its index) and using the characters in the word, 
    transforms it to an embedding of a fixed size.

    Args:
        char_vector: Pretrained character vectors. (maybe one-hot. need to verify this)
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
        num_filters: dimension of the output embeddings for each word.
    """
    
    def __init__(self, char_vectors, drop_prob, num_filters) -> None:
        super(_CharEmbedding, self).__init__()
        
        self.input_char_emb_size = char_vectors.size(1)
        self.num_filters = num_filters
        self.char_embed = nn.Embedding.from_pretrained(char_vectors, freeze = False) #output will be (batch_size, seq_length, chars_per_word, input_embedding_len)
        self.drop_prob = drop_prob

        self.conv1 = nn.Sequential(nn.Conv1d(in_channels = self.input_char_emb_size, out_channels =  self.num_filters, kernel_size = 3),# check dimensions passed here
                                nn.ReLU(),
                                nn.BatchNorm1d(num_features = self.num_filters),
                                # nn.Dropout(p = drop_prob),
                                nn.AdaptiveMaxPool1d(1)) # output will be (batch_size*seq_length, num_filters, 1)
        
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels = self.input_char_emb_size, out_channels =  self.num_filters, kernel_size = 5),# check dimensions passed here
                                nn.ReLU(),
                                nn.BatchNorm1d(num_features = self.num_filters),
                                # nn.Dropout(p = drop_prob),
                                nn.AdaptiveMaxPool1d(1)) # output will be (batch_size*seq_length, num_filters, 1)

    def forward(self, char_idxs):

        (batch_size, seq_len, _) = char_idxs.shape
        char_idxs = char_idxs.reshape(batch_size*seq_len, -1)
        
        emb = self.char_embed(char_idxs)
        emb = F.dropout(emb, self.drop_prob, self.training)

        emb = torch.transpose(emb, 1, 2)
        
        emb1 = self.conv1(emb)
        emb1 = torch.squeeze(emb1, dim=2)
        
        emb2= self.conv2(emb)
        emb2 = torch.squeeze(emb2, dim=2)

        emb = torch.cat((emb1, emb2), dim=1)

        # assert(emb.shape == (batch_size*seq_len, self.num_filters, 1))
        #emb = torch.squeeze(emb, dim=2)
        emb = emb.reshape(batch_size, seq_len, -1)

        assert(emb.shape == (batch_size, seq_len, self.num_filters *2))

        return emb
    
    def GetCharEmbedDim(self):
        return self.num_filters *2

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
       self.resizer = Initialized_Conv1d(self.word_embed_size + self.char_embed_dim, D, bias=False)
       self.hwy = HighwayEncoder(2, D)

   def forward(self, word_idxs, char_idxs):
       word_emb = self.word_embed(word_idxs)
       char_emb = self.char_embed(char_idxs)

       (batch_size, seq_len, _) = word_emb.shape
       assert(char_emb.shape == (batch_size, seq_len, self.char_embed_dim))

       word_emb = F.dropout(word_emb, self.drop_prob, self.training)

       emb = torch.cat((word_emb, char_emb), dim = 2)
       emb = emb.transpose(1,2)
       emb = self.resizer(emb)
       emb = self.hwy(emb)

       assert(emb.shape == (batch_size, self.GetOutputEmbeddingDim(), seq_len))
       return emb

   def GetOutputEmbeddingDim(self):
       return D

# 4. Model Encoder Layer. Similar to Seo et al. (2016)
# , the input of this layer at each position is [c, a, c ⊙ a, c ⊙ b], where a and b are
# respectively a row of attention matrix A and B. The layer parameters are the same as the Embedding
# Encoder Layer except that convolution layer number is 2 within a block and the total number of blocks are
# 7. We share weights between each of the 3 repetitions of the model encoder.
class EncoderBlock(nn.Module):
    def __init__(self, conv_num: int, ch_num: int, k: int, n_head=8):
        super().__init__()
        self.convs = nn.ModuleList([
            DepthwiseSeparableConv(ch_num, ch_num, k) for _ in range(conv_num)
        ])
        self.self_att = SelfAttention(n_head)
        self.FFN_1 = Initialized_Conv1d(ch_num, ch_num, relu=True, bias=True)
        self.FFN_2 = Initialized_Conv1d(ch_num, ch_num, bias=True)
        self.norm_C = nn.ModuleList([nn.LayerNorm(D) for _ in range(conv_num)])
        self.norm_1 = nn.LayerNorm(D)
        self.norm_2 = nn.LayerNorm(D)
        self.conv_num = conv_num
        #    def __init__(self, d_model, num_filters, kernel_size, num_conv_layers, num_heads, drop_prob=0.2):
        #        super(Encoder, self).__init__()
        #
        #        self.num_filters = num_filters
        #        self.positional_encoder = PositionalEncoder(d_model)
        #        self.drop_prob = drop_prob
        #        self.num_conv_layers = num_conv_layers
        #
        #        #depthwise separable conv
        #        self.conv_layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_conv_layers)])
        #        self.conv_layers = nn.ModuleList([DepthwiseSeparableConv(d_model, num_filters, kernel_size) for _ in range(num_conv_layers)])
        #
        #        #self attention
        #        self.att_layer_norm = nn.LayerNorm(d_model)
        #        self.att = SelfAttention(d_model, num_heads)
        #
        #        #feed-forward-layers
        #        self.ffn_layer_norm = nn.LayerNorm(num_filters)
        #        self.fc = nn.Linear(num_filters, num_filters, bias = True)

    def forward(self, x, mask, l, blks):
        total_layers = (self.conv_num+1)*blks
        out = PosEncoder(x)
        for i, conv in enumerate(self.convs):
            res = out
            out = self.norm_C[i](out.transpose(1,2)).transpose(1,2)
            if (i) % 2 == 0:
                out = F.dropout(out, p=dropout, training=self.training)
            out = conv(out)
            out = self.layer_dropout(out, res, dropout*float(l)/total_layers)
            l += 1
        res = out
        out = self.norm_1(out.transpose(1,2)).transpose(1,2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.self_att(out, mask)
        out = self.layer_dropout(out, res, dropout*float(l)/total_layers)
        l += 1
        res = out

        out = self.norm_2(out.transpose(1,2)).transpose(1,2)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.FFN_1(out)
        out = self.FFN_2(out)
        out = self.layer_dropout(out, res, dropout*float(l)/total_layers)
        return out

    #def forward(self, x):
#    (batch_size, seq_len, d_model) = x.shape
#
#    # out = self.positional_encoder(x)
#        out = x
#        # print('Positional Encoding shape is', out.shape)
#        assert (out.size() == (batch_size, seq_len, d_model))
#        #print("Input embedding is", x[0][5][:10])
#        #print("Positional embedding is", out[0][5][:10])
#
#        #out = out.add(x)
#        #print("Out embedding is", out[0][5][:10])
#        for i, conv_layer in enumerate(self.conv_layers):
#            res = out
#            out = self.conv_layer_norms[i](out)
#            out = torch.transpose(out, 1, 2)
#            out = conv_layer(out)
#            out = torch.transpose(out, 1, 2)
#            out = F.relu(out)
#            out = out + res
#            if (i + 1) % 2 == 0:
#                p_drop = self.drop_prob * (i + 1) / self.num_conv_layers
#                out = F.dropout(out, p=p_drop, training=self.training)
#
#        res = out
#
#        # print("Output size after conv layers in encoder",out.size())
#        assert (out.size() == (batch_size, seq_len, self.num_filters))
#
#        out = self.att_layer_norm(out)
#        out = self.att(out)
#        out = out + res
#        out = F.dropout(out, p=self.drop_prob, training=self.training)
#        res = out
#
#        out = self.ffn_layer_norm(out)
#        out = self.fc(out)
#        out = F.relu(out)
#        out = out + res
#        out = F.dropout(out, p=self.drop_prob, training=self.training)
#
#        # print("Output size after fully connected layer",out.size())
#        assert (out.shape == (batch_size, seq_len, self.num_filters))
#        return out

    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0,1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, 
                                 training=self.training) + residual
        else:
            return inputs + residual

# 3. Context-Query Attention Layer. This module is standard in almost every previous reading comprehension models such as Weissenborn et al. (2017) and Chen et al. (2017). We use C and Q to denote the encoded context and query.
# The context-to-query attention is constructed as follows: We first computer the similarities between each pair of context and query words, rendering a similarity matrix S ∈ Rn×m. We then normalize each row of S by applying the
# softmax function, getting a matrixS.Thenthecontext-to-queryattentioniscomputedasA=S·QT ∈Rn×d.Thesimilarity function used here is the trilinear function (Seo et al., 2016):
# f(q, c) = W0[q, c, q ⊙ c],
# where ⊙ is the element-wise multiplication and W0 is a trainable variable.
# Most high performing models additionally use some form of query-to-context attention, such as BiDaF (Seo et al., 2016) and DCN (Xiong et al., 2016). Empirically,
# we find that, the DCN attention can provide a little benefit over simply applying context-to-query attention, so we adopt this strategy.
# More concretely, we compute the column normalized matrix S of S by softmax function, and the
# query-to-context attention is B = S · S CT .
class CQAttention(nn.Module):
    def __init__(self):
        super().__init__()
        w4C = torch.empty(D, 1)
        w4Q = torch.empty(D, 1)
        w4mlu = torch.empty(1, 1, D)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, C, Q, Cmask, Qmask):
        C = C.transpose(1, 2) # after transpose (bs, ctxt_len, d_model)
        Q = Q.transpose(1, 2)# after transpose (bs, ques_len, d_model)
        batch_size = C.shape[0]
        Lc, Lq = C.shape[1], Q.shape[1]
        S = self.trilinear_for_attention(C, Q)
        Cmask = Cmask.view(batch_size, Lc, 1)
        Qmask = Qmask.view(batch_size, 1, Lq)
        S1 = F.softmax(mask_logits(S, Qmask), dim=2)
        S2 = F.softmax(mask_logits(S, Cmask), dim=1)
        # (bs, ctxt_len, ques_len) x (bs, ques_len, d_model) => (bs, ctxt_len, d_model)
        A = torch.bmm(S1, Q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2) # (bs, ctxt_len, 4 * d_model)
        return out.transpose(1, 2)

    def trilinear_for_attention(self, C, Q):
        C = F.dropout(C, p=dropout, training=self.training)
        Q = F.dropout(Q, p=dropout, training=self.training)
        Lc, Lq = C.shape[1], Q.shape[1]
        subres0 = torch.matmul(C, self.w4C).expand([-1, -1, Lq])
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, Lc, -1])
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1,2))
        # print('qanet_modules', subres0.shape, subres1.shape, subres2.shape)
        res = subres0 + subres1 + subres2
        res += self.bias
        return res

class QANetOutput(nn.Module):
   """Output layer used by QANet for question answering.

   As mentioned in the paper, output size of the encoding layers is (hidden_size = 128)
   They are basically the query informed context words representations

   Args:
       hidden_size (int): Hidden size used in the BiDAF model.
       drop_prob (float): Probability of zero-ing out activations.
   """
   def __init__(self, d_model):
       super(QANetOutput, self).__init__()
       self.start_linear = Initialized_Conv1d(2*d_model, 1)
       self.end_linear = Initialized_Conv1d(2*d_model,1 )

   def forward(self, m0, m1, m2, mask):

       (batch_size, d_model, seq_len) = m0.shape

       # (batch_size, seq_len, hidden_size)
       start_enc = torch.cat((m0, m1), dim =1)
       end_enc = torch.cat((m0, m2), dim = 1)

       assert(start_enc.shape == (batch_size, 2*d_model, seq_len))
       assert(end_enc.shape == (batch_size, 2*d_model, seq_len))

       # Shapes: (batch_size, seq_len, 1)
       logits_1 = self.start_linear(start_enc)
       logits_2 = self.end_linear(end_enc)

       assert(logits_1.shape == (batch_size, 1, seq_len))
       assert(logits_2.shape == (batch_size, 1, seq_len))

       # Shapes: (batch_size, seq_len)
       log_p1 = masked_softmax(logits_1.squeeze(dim=1), mask, log_softmax=True)
       log_p2 = masked_softmax(logits_2.squeeze(dim=1), mask, log_softmax=True)

       return log_p1, log_p2

 

