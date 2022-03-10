import torch
import torch.nn as nn
import torch.nn.functional
import math

import args



#The dimension of connectors in Qa-Network is used to be 128.
# layers_qa_net_d_model = args.get_train_args().d_model
# 4. Model Encoder Layer. Similar to Seo et al. (2016)
# the input of this layer at each position is [c, a, c ⊙ a, c ⊙ b], where a and b are
# respectively a row of attention matrix A and B. The layer parameters are the same as the Embedding
# Encoder Layer except that convolution layer number is 2 within a block and the total number of blocks are
# 7. We share weights between each of the 3 repetitions of the model encoder.
class the_encoder_block_layer(nn.Module):
    def __init__(self, number_of_convolutions, number_of_characters, k: int, n_head=8):
        super().__init__()
        self.norm_1 = nn.LayerNorm(128)
        self.norm_2 = nn.LayerNorm(128)
        list_of_submodules = [nn.LayerNorm(128) for index in range(number_of_convolutions)]
        self.convolutional_norm = nn.ModuleList(list_of_submodules)
        self.convs = nn.ModuleList([the_depth_wise_separable_convolution(number_of_characters, number_of_characters, k) for _ in range(number_of_convolutions)])
        self.self_att = the_self_attention_layer(n_head)
        self.FFN_1 = initializing_convolutional_oneD_layer(number_of_characters, number_of_characters, relu=True, bias=True)
        self.FFN_2 = initializing_convolutional_oneD_layer(number_of_characters, number_of_characters, bias=True)
        self.conv_num = number_of_convolutions

    def forward(self, x, mask, l, blks):
        total_layers = (self.conv_num+1)*blks
        out = PosEncoder(x)
        for i, conv in enumerate(self.convs):
            res = out
            out = self.convolutional_norm[i](out.transpose(1,2)).transpose(1,2)
            if (i) % 2 == 0:
                out = torch.nn.functional.dropout(out, p=0.1, training=self.training)
            out = conv(out)
            out = self.layer_dropout(out, res, 0.1*float(l)/total_layers)
            l += 1
        res = out
        out = self.norm_1(out.transpose(1,2)).transpose(1,2)
        out = torch.nn.functional.dropout(out, p=0.1, training=self.training)
        out = self.self_att(out, mask)
        out = self.layer_dropout(out, res, 0.1*float(l)/total_layers)
        l += 1
        res = out

        out = self.norm_2(out.transpose(1,2)).transpose(1,2)
        out = torch.nn.functional.dropout(out, p=0.1, training=self.training)
        out = self.FFN_1(out)
        out = self.FFN_2(out)
        out = self.layer_dropout(out, res, 0.1*float(l)/total_layers)
        return out

    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0,1) < dropout
            if pred:
                return residual
            else:
                return torch.nn.functional.dropout(inputs, dropout,
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
class the_context_query_attention_layer(nn.Module):
    def __init__(self):
        super().__init__()
        w4C = torch.empty(128, 1)
        w4Q = torch.empty(128, 1)
        w4mlu = torch.empty(1, 1,  128)
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
        S1 = nn.functional.softmax(mask_logits(S, Qmask), dim=2)
        S2 = nn.functional.softmax(mask_logits(S, Cmask), dim=1)
        # (bs, ctxt_len, ques_len) x (bs, ques_len, d_model) => (bs, ctxt_len, d_model)
        A = torch.bmm(S1, Q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2) # (bs, ctxt_len, 4 * d_model)
        return out.transpose(1, 2)

    def trilinear_for_attention(self, C, Q):
        C = torch.nn.functional.dropout(C, p=0.1, training=self.training)
        Q = torch.nn.functional.dropout(Q, p=0.1, training=self.training)
        Lc, Lq = C.shape[1], Q.shape[1]
        subres0 = torch.matmul(C, self.w4C).expand([-1, -1, Lq])
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, Lc, -1])
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1,2))
        # print('qanet_modules', subres0.shape, subres1.shape, subres2.shape)
        res = subres0 + subres1 + subres2
        res += self.bias
        return res

class initializing_convolutional_oneD_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, relu=False, 
                 stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.out = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, groups=groups, bias=bias)
        if relu is True:
            self.relu = True
            # Note on Kaiming normal: https://towardsdatascience.com/understand-kaiming-initialization-and-implementation-detail-in-pytorch-f7aa967e9138
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            self.relu = False
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        if self.relu == True:
            return nn.functional.relu(self.out(x))
        else:
            return self.out(x)

# class PositionalEncoder(nn.Module):
#     # Reference: https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/positional_encodings.py
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




class the_high_way_network(nn.Module):
    def __init__(self, layer_num: int, size= 128):
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([
            initializing_convolutional_oneD_layer(size, size, relu=False, bias=True)
            for _ in range(self.n)
        ])
        self.gate = nn.ModuleList([
            initializing_convolutional_oneD_layer(size, size, bias=True) for _ in range(self.n)])

    def forward(self, x):
        #x: shape [batch_size, hidden_size, length]
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = self.linear[i](x)
            nonlinear = torch.nn.functional.dropout(nonlinear, p=0.1, training=self.training)
            x = gate * nonlinear + (1 - gate) * x
            #x = F.relu(x)
        return x
    #
    # for gate, transform in zip(self.gates, self.transforms):
    #     # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
    #     g = torch.sigmoid(gate(x))
    #     t = F.relu(transform(x))
    #     x = g * t + (1 - g) * x


#class DepthwiseSeparableConv(nn.Module):
#    def __init__(self, in_channels, out_channels, kernel_size):
#
#        # Can refernce this from - https://github.com/heliumsea/QANet-pytorch/blob/master/models.py#L39
#        super(DepthwiseSeparableConv, self).__init__()
#        self.depthwiseConv = nn.Conv1d(in_channels, in_channels, kernel_size, padding = kernel_size//2, groups = in_channels)
#        self.pointwiseConv = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding = 0)
#    #def forward(self, x):
#        out = self.pointwiseConv(self.depthwiseConv(x))
#        return out
class the_depth_wise_separable_convolution(nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(
            in_channels=in_ch, out_channels=in_ch, kernel_size=k, 
            groups=in_ch, padding=k // 2, bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, 
                                        kernel_size=1, padding=0, bias=bias)
    def forward(self, x):
        return nn.functional.relu(self.pointwise_conv(self.depthwise_conv(x)))

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
class the_embedding_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.high_way_network = the_high_way_network(2)
        self.dim1, self.dim2, self.dim3, self.dim4 = 0, 3, 1, 2
        # print("in channels: ", args.get_train_args().char_dim)
        # print("out channels: ", args.get_train_args().d_model)
        #Applies a 2D convolution layer over an input signal composed of several input planes.
        #The output is of the format: batch size, number of channels, height of input planes in pixels, and width in pixels.
        #Note on Kaiming normal: https://towardsdatascience.com/understand-kaiming-initialization-and-implementation-detail-in-pytorch-f7aa967e9138
        #Use He initialization
        self.one_d_convolution = initializing_convolutional_oneD_layer(args.get_train_args().glove_dim+ 128, 128, bias=False)
        self.two_d_convolution = nn.Conv2d(in_channels=args.get_train_args().char_dim,  out_channels=128, kernel_size=(1,5), padding=0, bias=True)
        nn.init.kaiming_normal_(self.two_d_convolution.weight, nonlinearity='relu')

    def forward(self, character_level_embedding, word_level_embedding):
        #During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution.
        word_level_embedding = torch.nn.functional.dropout(word_level_embedding, p=0.1, training=self.training)
        word_level_embedding = word_level_embedding.transpose(1, 2)
        intermediate1 = character_level_embedding.permute(self.dim1, self.dim2, self.dim3, self.dim4)
        intermediate2 = torch.nn.functional.dropout(intermediate1, p=args.get_train_args().qanet_char_dropout, training=self.training)
        intermediate3 = torch.nn.functional.relu(self.two_d_convolution(intermediate2))
        intermediate4, _ = torch.max(intermediate3, dim=3)
        character_level_embedding = intermediate4.squeeze()
        #Concatenating the word level embedding and the character-level embedding
        total_embedding = torch.cat([character_level_embedding, word_level_embedding], dim=1)
        total_embedding = self.one_d_convolution(total_embedding)
        total_embedding = self.high_way_network(total_embedding)
        return total_embedding

class the_self_attention_layer(nn.Module):
    def __init__(self, n_head=8):
        super().__init__()
        # self.n_head = n_head
        # self.n_embed = n_embed
        self.n_head = n_head
        self.mem_conv = initializing_convolutional_oneD_layer(
            128,  128 * 2, kernel_size=1, relu=False, bias=False)
        self.query_conv = initializing_convolutional_oneD_layer(
            128,  128, kernel_size=1, relu=False, bias=False)
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
            for tensor in torch.split(memory,  128, dim=2)
        ]
        # att = att.masked_fill(self.mask[:,:,:seq_len,:seq_len] == 0, -1e10) # todo: just use float('-inf') instead?
        # att = F.softmax(att, dim=-1)
        # att = self.attn_drop(att)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # y = y.transpose(1, 2).contiguous().view(B, seq_len, C) # re-assemble all head outputs side by side
        # y = self.resid_drop(self.proj(y))
        # return y


        key_depth_per_head =  128 // self.n_head
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
        weights = torch.nn.functional.softmax(logits, dim=-1)
        # dropping out the attention links for each of the heads
        weights = torch.nn.functional.dropout(weights, p=0.1, training=self.training)
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


class Pointer(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = initializing_convolutional_oneD_layer( 128*2, 1)
        self.w2 = initializing_convolutional_oneD_layer( 128*2, 1)

    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat([M1, M2], dim=1)
        X2 = torch.cat([M1, M3], dim=1)
        Y1 = mask_logits(self.w1(X1).squeeze(), mask)
        Y2 = mask_logits(self.w2(X2).squeeze(), mask)
        # print(Y1[0])
        p1 = nn.functional.log_softmax(Y1, dim=1)
        p2 = nn.functional.log_softmax(Y2, dim=1)
        return p1, p2


#class QANetOutput(nn.Module):
#    """Output layer used by QANet for question answering.
#
#    As mentioned in the paper, output size of the encoding layers is (hidden_size = 128)
#    They are basically the query informed context words representations
#
#    Args:
#        hidden_size (int): Hidden size used in the BiDAF model.
#        drop_prob (float): Probability of zero-ing out activations.
#    """
#    def __init__(self, d_model, drop_prob):
#        super(QANetOutput, self).__init__()
#        self.start_linear = nn.Linear(2*d_model, 1, bias = False)
#        self.end_linear = nn.Linear(2*d_model,1 , bias = False)
#
#    def forward(self, m0, m1, m2, mask):
#
#        (batch_size, seq_len, d_model) = m0.shape
#
#        # (batch_size, seq_len, hidden_size)
#        start_enc = torch.cat((m0, m1), dim =2)
#        end_enc = torch.cat((m0, m2), dim = 2)
#
#        assert(start_enc.shape == (batch_size, seq_len, 2*d_model))
#        assert(end_enc.shape == (batch_size, seq_len, 2*d_model))
#
#        # Shapes: (batch_size, seq_len, 1)
#        logits_1 = self.start_linear(start_enc)
#        logits_2 = self.end_linear(end_enc)
#
#        assert(logits_1.shape == (batch_size, seq_len, 1))
#        assert(logits_2.shape == (batch_size, seq_len, 1))
#
#        # Shapes: (batch_size, seq_len)
#        log_p1 = masked_softmax(logits_1.squeeze(dim=2), mask, log_softmax=True)
#        log_p2 = masked_softmax(logits_2.squeeze(dim=2), mask, log_softmax=True)
#
#        return log_p1, log_p2



def mask_logits(inputs, mask):
    mask = mask.type(torch.float32)
    return inputs + (-1e30) * (1 - mask)
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