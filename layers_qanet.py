import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import args
from util import masked_softmax

train_args = args.get_train_args()
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
       self.start_linear = FeedForwardHelper(2*d_model, 1)
       self.end_linear = FeedForwardHelper(2*d_model, 1)

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




# class PositionalEncoder(nn.Module):
#     def __init__(self, in_channels):
#         super(PositionalEncoder, self).__init__()
#         if in_channels % 2 == 0:
#             self.channels = in_channels
#         else:
#             self.channels = in_channels + 1
#         self.frequency_factor = 1.0 / (10000 ** (torch.arange(0, self.channels, 2).float() / self.channels))
#
#     def forward(self, tensor):
#
#         batch_size, x, orig_ch = tensor.shape
#         # print("positional encoding orig shape", tensor.shape)
#         pos_x = torch.arange(x, device=tensor.device).type(self.frequency_factor.type())
#         sin_inp_x = torch.einsum("i,j->ij", pos_x, self.frequency_factor)
#         # print("sin_inp_x apply sincos", sin_inp_x.shape)
#         emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
#         # print("embx apply sincos", emb_x.shape)
#         emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
#         # print("emb zeros", emb.shape)
#         emb[:, : self.channels] = emb_x
#         # print("output", emb[None, :, :orig_ch].repeat(batch_size, 1, 1).shape)
#         return emb[None, :, :orig_ch].repeat(batch_size, 1, 1)

#A positional encoding is added to the input at the beginning of each encoder layer consisting of sin and cos functions at varying wavelengths,
# as defined in (Vaswani et al., 2017a). Each sub-layer after the positional encoding (one of convolution, self-attention, or feed-forward-net)
# inside the encoder structure is wrapped inside a residual block. (Page 3 of the paper).
def PositionalEncoder(thetensor, minimum=1.0, maximum=1.0e4):
    thetensor = thetensor.transpose(1, 2)
    len, channel_number = thetensor.shape[1], thetensor.shape[2]
    position = torch.arange(len).type(torch.float32)
    number_scale = channel_number // 2
    log_scale_incrementing = (math.log(float(maximum) / float(minimum)) / (float(number_scale)-1))
    inverted_scale = minimum * torch.exp(torch.arange(number_scale).type(torch.float32) * -log_scale_incrementing)
    scaled = position.unsqueeze(1) * inverted_scale.unsqueeze(0)
    sig = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim = 1)
    m = nn.ZeroPad2d((0, (channel_number % 2), 0, 0))
    sig = m(sig)
    sig = sig.view(1, len, channel_number)
    if torch.cuda.is_available():
        sig = sig.cuda()
    return (thetensor + sig).transpose(1,2)


class HighwayEncoder(nn.Module):
    def __init__(self, layer_num: int, size=train_args.d_model):
        super().__init__()
        self.n = layer_num
        self.transforms = nn.ModuleList([FeedForwardHelper(size, size, relu=False, bias=True)for index in range(self.n)])
        self.gate = nn.ModuleList([FeedForwardHelper(size, size, bias=True) for _ in range(self.n)])

    def forward(self, x):
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = self.transforms[i](x)
            nonlinear = F.dropout(nonlinear, p=train_args.qanet_dropout, training=self.training)
            x = gate * nonlinear + (1 - gate) * x
        return x

class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, k, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=k, groups=in_channels, padding=k // 2, bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=1, padding=0, bias=bias)
    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))


class SelfAttention(nn.Module):
    def __init__(self, number_of_heads=8):
        super().__init__()
        # self.n_head = n_head
        # self.n_embed = n_embed
        self.number_of_heads = number_of_heads
        self.convolution1 = FeedForwardHelper(train_args.d_model, train_args.d_model * 2, kernel_size=1, relu=False, bias=False)
        self.query_conv = FeedForwardHelper(train_args.d_model, train_args.d_model, kernel_size=1, relu=False, bias=False)
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

    def forward(self, query, mask):
        temp = query
        # B, seq_len, C = x.size() #64, 287, 128
        temp = self.convolution1(temp)
        query = self.query_conv(query)
        temp = temp.transpose(1, 2)
        query = query.transpose(1, 2)
        # # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # k = self.key(x).view(B, seq_len, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # q = self.query(x).view(B, seq_len, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # v = self.value(x).view(B, seq_len, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.helper(query, self.number_of_heads)
        k, v = [self.helper(tensor, self.number_of_heads)for tensor in torch.split(temp, train_args.d_model, dim=2)]
        # att = att.masked_fill(self.mask[:,:,:seq_len,:seq_len] == 0, -1e10) # todo: just use float('-inf') instead?
        # att = F.softmax(att, dim=-1)
        # att = self.attn_drop(att)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # y = y.transpose(1, 2).contiguous().view(B, seq_len, C) # re-assemble all head outputs side by side
        # y = self.resid_drop(self.proj(y))
        # return y
        q *= (train_args.d_model // self.number_of_heads)**(-0.5)
        bias = False
        intermediary = torch.matmul(q,k.permute(0,1,3,2))
        if bias:
            intermediary += self.bias
        if mask is not None:
            shapes = [number if number != None else -1 for number in list(intermediary.size())]
            mask = mask.view(shapes[0], 1, 1, shapes[-1])
            mask = mask.type(torch.float32)
            logits = intermediary + (-1e30) * (1 - mask)
        weights = F.softmax(logits, dim=-1)
        weights = F.dropout(weights, p=train_args.qanet_dropout, training=self.training)
        intermediary =  torch.matmul(weights, v)
        toReturn = (intermediary).permute(0,2,1,3)
        s_value = list(toReturn.size())
        new_shape = s_value[:-2] + [ s_value[-2:][0] *  s_value[-2:][1] if  s_value[-2:][0] and  s_value[-2:][1] else None]
        return toReturn.contiguous().view(new_shape).transpose(1, 2)

    def helper(self, input, number):
        temp_s = list(input.size())
        last = temp_s[-1]
        shape = temp_s[:-1] + [number] + [last // number if last else None]
        result = input.view(shape).permute(0, 2, 1, 3)
        return result

# class Embedding(nn.Module):
#     """Embedding layer used by BiDAF, without the character-level component.
#
#     Word-level embeddings are further refined using a 2-layer Highway Encoder
#     (see `HighwayEncoder` class for details).
#
#     Args:
#         word_vectors (torch.Tensor): Pre-trained word vectors.
#         hidden_size (int): Size of hidden activations.
#         drop_prob (float): Probability of zero-ing out activations
#     """
#     def __init__(self, word_vectors, hidden_size, drop_prob):
#         super(Embedding, self).__init__()
#         self.drop_prob = drop_prob
#         self.embed = nn.Embedding.from_pretrained(word_vectors)
#
#         # word_vectors include te word_vectors for all words in vocab
#         # word_vectors.size = (num_vocab, embed_size)
#         self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
#         # Projection layer has size (embed_size, hidden_size)
#         self.hwy = HighwayEncoder(2, hidden_size)
#
#     def forward(self, x):
#         emb = self.embed(x)   # (batch_size, seq_len, embed_size)
#         emb = F.dropout(emb, self.drop_prob, self.training)
#         emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
#         emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)
#
#         return emb


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
        self.conv2d = nn.Conv2d(train_args.char_dim, train_args.d_model, kernel_size=(1,5), padding=0, bias=True)
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')
        self.conv1d = FeedForwardHelper(train_args.glove_dim+train_args.d_model, train_args.d_model, bias=False)
        self.highwayEncoder = HighwayEncoder(2)

    def forward(self, character_level_emb, word_level_emb):
        word_level_emb = F.dropout(word_level_emb, p=train_args.qanet_dropout, training=self.training)
        word_level_emb = word_level_emb.transpose(1, 2)
        character_level_emb = character_level_emb.permute(0, 3, 1, 2)
        character_level_emb = F.dropout(character_level_emb, p=train_args.qanet_char_dropout, training=self.training)
        character_level_emb = self.conv2d(character_level_emb)
        character_level_emb = F.relu(character_level_emb)
        character_level_emb, _ = torch.max(character_level_emb, dim=3)
        ch_emb = character_level_emb.squeeze()
        emb = self.conv1d(torch.cat([ch_emb, word_level_emb], dim=1))
        return self.highwayEncoder(emb)
#class QANetEmbedding(nn.Module):
#    """Combines the Word and Character embedding and then applies a transformation and highway network.
#    Output of this layer will be (batch_size, seq_len, hidden_size)
#    """
#
#    def __init__(self, word_vectors, char_vectors, drop_prob, num_filters):
#        super(QANetEmbedding, self).__init__()
#        self.drop_prob = drop_prob
#        self.word_embed_size = word_vectors.size(1)
#        self.batch_size = word_vectors.size(0)
#
#        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
#        self.char_embed = _CharEmbedding(char_vectors=char_vectors, drop_prob=drop_prob, num_filters = num_filters)
#        self.char_embed_dim = self.char_embed.GetCharEmbedDim()
#
#        self.hwy = HighwayEncoder(2, self.char_embed_dim + self.word_embed_size)
#
#    def forward(self, word_idxs, char_idxs):
#        word_emb = self.word_embed(word_idxs)
#        char_emb = self.char_embed(char_idxs)
#
#        (batch_size, seq_len, _) = word_emb.shape
#        assert(char_emb.shape == (batch_size, seq_len, self.char_embed_dim))
#
#        word_emb = F.dropout(word_emb, self.drop_prob, self.training)
#
#        emb = torch.cat((word_emb, char_emb), dim = 2)
#        emb = self.hwy(emb)
#
#        assert(emb.shape == (batch_size, seq_len, self.GetOutputEmbeddingDim()))
#        return emb
#
#    def GetOutputEmbeddingDim(self):
#        return self.word_embed_size + self.char_embed_dim

# 4. Model Encoder Layer. Similar to Seo et al. (2016)
# , the input of this layer at each position is [c, a, c ⊙ a, c ⊙ b], where a and b are
# respectively a row of attention matrix A and B. The layer parameters are the same as the Embedding
# Encoder Layer except that convolution layer number is 2 within a block and the total number of blocks are
# 7. We share weights between each of the 3 repetitions of the model encoder.
class theEncoderblock(nn.Module):
    def __init__(self, number_of_convolutions: int, number_of_characters: int, k: int, number_of_head=8):
        super().__init__()
        self.number_of_convolutions = number_of_convolutions
        self.self_attention = SelfAttention(number_of_head)
        self.layer_norm_1 = nn.LayerNorm(train_args.d_model)
        self.layer_norm_2 = nn.LayerNorm(train_args.d_model)
        self.convolutions = nn.ModuleList([DepthwiseSeparableConvolution(number_of_characters, number_of_characters, k) for _ in range(number_of_convolutions)])
        self.feedforwardnetwork1 = FeedForwardHelper(number_of_characters, number_of_characters, relu=True, bias=True)
        self.feedforwardnetwork2 = FeedForwardHelper(number_of_characters, number_of_characters, bias=True)
        self.norm_C = nn.ModuleList([nn.LayerNorm(train_args.d_model) for _ in range(number_of_convolutions)])

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

    def forward(self, inputx, mask, number_of_layers, blocks):
        layers = (self.number_of_convolutions+1)*blocks
        result = PositionalEncoder(inputx)
        for index, convolution in enumerate(self.convolutions):
            res = result
            result = self.norm_C[index](result.transpose(1,2)).transpose(1,2)
            if (index) % 2 == 0:
                result = F.dropout(result, p=train_args.qanet_dropout, training=self.training)
            result = convolution(result)
            input = result
            residual = res
            dropout_probability = train_args.qanet_dropout*float(number_of_layers)/layers
            if self.training:
                if torch.empty(1).uniform_(0, 1) < dropout_probability:
                    result = residual
                else:
                    result = F.dropout(input, dropout_probability, training=self.training) + residual
            else:
                result = input + residual
            number_of_layers += 1
        res = result
        result = self.layer_norm_1(result.transpose(1,2)).transpose(1,2)
        result = F.dropout(result, p=train_args.qanet_dropout, training=self.training)
        result = self.self_attention(result, mask)

        input = result
        residual = res
        dropout_probability = train_args.qanet_dropout*float(number_of_layers)/layers
        if self.training:
            if torch.empty(1).uniform_(0, 1) < dropout_probability:
                result = residual
            else:
                result = F.dropout(input, dropout_probability, training=self.training) + residual
        else:
            result = input + residual
        number_of_layers += 1
        res = result

        result = self.layer_norm_2(result.transpose(1,2)).transpose(1,2)
        result = F.dropout(result, p=train_args.qanet_dropout, training=self.training)
        result = self.feedforwardnetwork1(result)
        result = self.feedforwardnetwork2(result)
        input = result
        residual = res
        dropout_probability = train_args.qanet_dropout*float(number_of_layers)/layers
        if self.training:
            if torch.empty(1).uniform_(0, 1) < dropout_probability:
                result = residual
            else:
                result = F.dropout(input, dropout_probability, training=self.training) + residual
        else:
            result = input + residual
        return result

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



# 3. Context-Query Attention Layer. This module is standard in almost every previous reading comprehension models such as Weissenborn et al. (2017) and Chen et al. (2017). We use C and Q to denote the encoded context and query.
# The context-to-query attention is constructed as follows: We first computer the similarities between each pair of context and query words, rendering a similarity matrix S ∈ Rn×m. We then normalize each row of S by applying the
# softmax function, getting a matrixS.Thenthecontext-to-queryattentioniscomputedasA=S·QT ∈Rn×d.Thesimilarity function used here is the trilinear function (Seo et al., 2016):
# f(q, c) = W0[q, c, q ⊙ c],
# where ⊙ is the element-wise multiplication and W0 is a trainable variable.
# Most high performing models additionally use some form of query-to-context attention, such as BiDaF (Seo et al., 2016) and DCN (Xiong et al., 2016). Empirically,
# we find that, the DCN attention can provide a little benefit over simply applying context-to-query attention, so we adopt this strategy.
# More concretely, we compute the column normalized matrix S of S by softmax function, and the
# query-to-context attention is B = S · S CT .
# Same as BiDAF Attention
class ContextQueryAttention(nn.Module):
    """ContextQueryAttention attention originally used by BiDAF.

    ContextQueryAttention attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self):
        super().__init__()
        c_weight = torch.empty(train_args.d_model, 1)
        q_weight = torch.empty(train_args.d_model, 1)
        cq_weight = torch.empty(1, 1, train_args.d_model)
        nn.init.xavier_uniform_(c_weight)
        nn.init.xavier_uniform_(q_weight)
        nn.init.xavier_uniform_(cq_weight)
        self.c_weight = nn.Parameter(c_weight)
        self.q_weight = nn.Parameter(q_weight)
        self.cq_weight = nn.Parameter(cq_weight)
        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, c, q, c_mask, q_mask):
        # c is the context. Its size will be (batch_size, context_len, hidden_size) as this is coming from the hidden state of the RNN
        # q is the query. Its size will be (batch_size, query_len, hidden_size)
        c = c.transpose(1, 2)
        q = q.transpose(1, 2)
        batch_size = c.shape[0]
        length_c, length_q = c.shape[1], q.shape[1]
        s = self.get_similarity_matrix(c, q)
        c_mask = c_mask.view(batch_size, length_c, 1)
        q_mask = q_mask.view(batch_size, 1, length_q)
        mask1 = q_mask.type(torch.float32)
        intermediary_s1 = s + (-1e30) * (1 - mask1)
        mask2 = c_mask.type(torch.float32)
        intermediary_s2 = s + (-1e30) * (1 - mask2)
        s1 = F.softmax(intermediary_s1, dim=2)
        s2 = F.softmax(intermediary_s2, dim=1)
        a = torch.bmm(s1, q)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)
        out = torch.cat([c, a, torch.mul(c, a), torch.mul(c, b)], dim=2)
        out =  out.transpose(1, 2)
        return out



    def get_similarity_matrix(self, c, q):
        c = F.dropout(c, p=train_args.qanet_dropout, training=self.training)
        q = F.dropout(q, p=train_args.qanet_dropout, training=self.training)
        length_c, length_q = c.shape[1], q.shape[1]
        intermediary0 = torch.matmul(c, self.c_weight).expand([-1, -1, length_q])
        intermediary1 = torch.matmul(q, self.q_weight).transpose(1, 2).expand([-1, length_c, -1])
        intermediary2 = torch.matmul(c * self.cq_weight, q.transpose(1,2))
        result = intermediary0 + intermediary1 + intermediary2
        result += self.bias
        return result


# class BiDAFAttention(nn.Module):
#     """Bidirectional attention originally used by BiDAF.
#
#     Bidirectional attention computes attention in two directions:
#     The context attends to the query and the query attends to the context.
#     The output of this layer is the concatenation of [context, c2q_attention,
#     context * c2q_attention, context * q2c_attention]. This concatenation allows
#     the attention vector at each timestep, along with the embeddings from
#     previous layers, to flow through the attention layer to the modeling layer.
#     The output has shape (batch_size, context_len, 8 * hidden_size).
#
#     Args:
#         hidden_size (int): Size of hidden activations.
#         drop_prob (float): Probability of zero-ing out activations.
#     """
#     def __init__(self, hidden_size, drop_prob=0.1):
#         super(BiDAFAttention, self).__init__()
#         self.drop_prob = drop_prob
#         self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
#         self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
#         self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
#         for weight in (self.c_weight, self.q_weight, self.cq_weight):
#             nn.init.xavier_uniform_(weight)
#         self.bias = nn.Parameter(torch.zeros(1))
#
#     def forward(self, c, q, c_mask, q_mask):
#
#         # c is the context. Its size will be (batch_size, context_len, hidden_size) as this is coming from the hidden state of the RNN
#         # q is the query. Its size will be (batch_size, query_len, hidden_size)
#
#         batch_size, c_len, hidden_size = c.size()
#         q_len = q.size(1)
#         s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
#         c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
#         q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
#         s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len) #looks like context2query attention as we are setting the places where query has a pad element to 0 through the mask
#         s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)
#
#         # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
#         a = torch.bmm(s1, q) # a is final context to query attention
#         # a denotes the attention vector for each word of the context to the query
#         # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
#         b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)
#
#         x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)
#         assert(x.shape == (batch_size, c_len, 4 * hidden_size))
#         return x
#
#     def get_similarity_matrix(self, c, q):
#         """Get the "similarity matrix" between context and query (using the
#         terminology of the BiDAF paper).
#
#         A naive implementation as described in BiDAF would concatenate the
#         three vectors then project the result with a single weight matrix. This
#         method is a more memory-efficient implementation of the same operation.
#
#         See Also:
#             Equation 1 in https://arxiv.org/abs/1611.01603
#         """
#         c_len, q_len = c.size(1), q.size(1)
#         c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
#         q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)
#
#         # Shapes: (batch_size, c_len, q_len)
#         s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
#         s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
#                                            .expand([-1, c_len, -1])
#         s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
#         s = s0 + s1 + s2 + self.bias
#
#         return s



class FeedForwardHelper(nn.Module):
    def __init__(self, inc, outc, kernel_size=1, relu=False,stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.out = nn.Conv1d(inc, outc, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        if relu:
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
            self.relu = True
        else:
            nn.init.xavier_uniform_(self.out.weight)
            self.relu = False

    def forward(self, x):
        if self.relu:
            return F.relu(self.out(x))
        else:
            return self.out(x)