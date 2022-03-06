"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import layers_char_embed
import layers_qanet
import layers_qanet1
import layers1
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.): # added the char_vectors here just to make the init function same in both models
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs): # last two parameters are unused here at the moment
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out

class BiDAFWithChar(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        char_vectors (torch.Tensor): Pre-trained char vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.):
        super(BiDAFWithChar, self).__init__()

        self.char_embed_size = 100
        self.emb = layers_char_embed.BiDAFWordPlusCharEmbedding(word_vectors=word_vectors,
                                    char_vectors=char_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob,
                                    num_filters=self.char_embed_size)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        # to do : determine what do about the masking for the character embedding
        # answer: nothing needs to be done here

        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out

class QANet(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        char_vectors (torch.Tensor): Pre-trained char vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, drop_prob=0.):
        super(QANet, self).__init__()
        # self.word_embed_size = word_vectors.size(1)
        self.d_model = 128 # d model is the dimensionality of each word before and after it goes into the encoder layer, i
        self.num_conv_filters = 128

        self.emb = layers_qanet.QANetEmbedding(word_vectors=word_vectors,
                                    char_vectors=char_vectors,
                                    drop_prob=drop_prob,
                                    d_model= self.d_model)

        # self.initial_embed_dim = self.emb.GetOutputEmbeddingDim()
        

        # These two layers will reduce the dimensionality of the embedding of each word from (500) to (128)
        # self.context_conv = layers_qanet.DepthwiseSeparableConv(
        #     in_channels = self.initial_embed_dim, 
        #     out_channels= self.d_model,
        #     kernel_size=5)
        
        # self.question_conv = layers_qanet.DepthwiseSeparableConv(
        #     in_channels = self.initial_embed_dim, 
        #     out_channels = self.d_model,
        #     kernel_size=5)

        # Output of the Convolutions above will be fed into the encoder
        self.embedding_encoder_context =  layers_qanet.Encoder(d_model=self.d_model,
                                                                num_filters=self.num_conv_filters, 
                                                                kernel_size=7, 
                                                                num_conv_layers=4, 
                                                                num_heads=8, 
                                                                drop_prob=drop_prob)

        self.embedding_encoder_question =  layers_qanet.Encoder(d_model=self.d_model,
                                                                num_filters=self.num_conv_filters, 
                                                                kernel_size=7, 
                                                                num_conv_layers=4, 
                                                                num_heads=8, 
                                                                drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=self.d_model, drop_prob=drop_prob)

        self.att_conv = layers_qanet.DepthwiseSeparableConv(
            in_channels = self.d_model*4, 
            out_channels = self.d_model,
            kernel_size=5)

        self.model_encoders =  nn.ModuleList([layers_qanet.Encoder(d_model=self.d_model,
                                                                num_filters=self.num_conv_filters,
                                                                kernel_size=5,
                                                                num_conv_layers=2,
                                                                num_heads=8,
                                                                drop_prob=drop_prob) for _ in range(5)])
        

        self.out = layers_qanet.QANetOutput(d_model=self.d_model, drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)
        batch_size = cw_idxs.shape[0]

        # In QANet the projection is not applied and output of highway network is same size as the word+char embedding dim
        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, self.initial_embed_dim)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, self.initial_embed_dim)

        # print("context embedding shape", c_emb.shape)
        # print("query embedding shape", q_emb.shape)

        #c_emb = self.context_conv(c_emb.transpose(1,2)).transpose(1,2) # (batch_size, self.num_conv_filters, c_len)
        #q_emb = self.question_conv(q_emb.transpose(1,2)).transpose(1,2) # (batch_size, self.num_conv_filters, q_len)

        c_enc = self.embedding_encoder_context(c_emb)
        q_enc = self.embedding_encoder_question(q_emb)

        # print("context encoding", c_enc[0][5][:10])
        # print("query encoding shape",q_enc.shape)

        # assert(c_enc.shape == (batch_size, c_len, self.hidden_size))
        # assert(q_enc.shape == (batch_size, q_len, self.hidden_size))

        # compute attention same as BiDAF
        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 4 * d_model)

        # print("context-query attention shape", att.shape)
        # assert(att.shape == (batch_size, c_len, 8 * self.d_model))
        m0 = self.att_conv(att.transpose(1,2)).transpose(1,2)
        
        #print("m0", m0[0][5][:10])
        for i, enc in enumerate(self.model_encoders):
            m0 = enc(m0)
        m1 = m0

        for i, enc in enumerate(self.model_encoders):
            m0 = enc(m0)
        m2 = m0

        for i, enc in enumerate(self.model_encoders):
            m0 = enc(m0)
        m3 = m0

        out = self.out(m1, m2, m3, c_mask)

        return out
        
class QANet1(nn.Module):
    def __init__(self, word_vectors, char_vectors, char_vocab_size,drop_prob=0.):
        super(QANet1, self).__init__()
        # self.word_embed_size = word_vectors.size(1)
        self.d_model = 128 # d model is the dimensionality of each word before and after it goes into the encoder layer, i
        self.num_conv_filters = 128

        self.emb = Embedding(word_vectors=word_vectors,
                                    char_vocab_size=char_vocab_size,
                                    word_emb_size= 200,
                                    char_emb_size= 200,
                                    drop_prob_char=drop_prob,
                                    drop_prob_word = drop_prob)
                        
        self.context_conv = nn.Linear(500,self.d_model)
        # self.question_conv = layers_qanet1.DepthwiseSeparableConv(500,self.d_model, 5)
        self.emb_enc = layers1.Embedding_Encoder(num_blocks=1,
                                                 num_conv = 4,
                                                  kernel_size = 5,
                                                  hidden_size = self.d_model,
                                                  num_heads = 8,
                                                  survival_prob= 0.8)

        # self.c_emb_enc = layers_qanet1.EncoderBlock(conv_num=4, ch_num=self.d_model, k=7)
        # self.q_emb_enc = layers_qanet1.EncoderBlock(conv_num=4, ch_num=self.d_model, k=7)
        self.cq_att = layers_qanet1.CQAttention()
        self.cq_resizer = layers_qanet1.DepthwiseSeparableConv(self.d_model * 4, self.d_model, 5)

        self.mod = layers1.Model_Encoder(num_blocks = 5, 
                                        num_conv = 2, 
                                        kernel_size=5, 
                                        hidden_size = self.d_model, 
                                        num_heads = 8, 
                                        survival_prob= 0.8)
        
        # enc_blk = layers_qanet1.EncoderBlock(conv_num=2, ch_num=self.d_model, k=5)
        # self.model_enc_blks = nn.ModuleList([enc_blk] * 7)
        self.out = layers_qanet.QANetOutput(d_model=self.d_model, drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        cmask = torch.zeros_like(cw_idxs) != cw_idxs
        qmask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = cmask.sum(-1), qmask.sum(-1)
        batch_size = cw_idxs.shape[0]

        # Cw, Cc = self.word_emb(cw_idxs), self.char_emb(cc_idxs)
        # Qw, Qc = self.word_emb(qw_idxs), self.char_emb(qw_idxs)
        # C, Q = self.emb(Cc, Cw), self.emb(Qc, Qw)
        # c_emb = self.context_conv(C)  
        # q_emb = self.question_conv(Q)  

        # In QANet the projection is not applied and output of highway network is same size as the word+char embedding dim
        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, self.initial_embed_dim)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, self.initial_embed_dim)

        #print('c_emb before projection is', c_emb[0:5])
        c_emb = nn.functional.relu(self.context_conv(c_emb))
        q_emb = nn.functional.relu(self.context_conv(q_emb))
        #print('c_emb after projection is', c_emb[0:5])
        
        #c_emb = c_emb.transpose(1,2)
        #q_emb = q_emb.transpose(1,2)

        Ce, Qe = self.emb_enc(c_emb, q_emb, cmask, qmask)

        #Ce = self.c_emb_enc(c_emb, cmask)
        # print('Mean of context encdoing is', torch.mean(Ce))
        #print('c_enc is', Ce.transpose(1,2)[0:5])

        #Qe = self.q_emb_enc(q_emb, qmask)
        Ce = Ce.transpose(1,2)
        Qe = Qe.transpose(1,2)
        att = self.cq_att(Ce, Qe, cmask, qmask)
        att = self.cq_resizer(att)

        att = att.transpose(1,2)
        mod1, mod2, mod3 = self.mod(att, cmask)

        # for enc in self.model_enc_blks: M1 = enc(M1, cmask)
        # M2 = M1
        # for enc in self.model_enc_blks: M2 = enc(M2, cmask)
        # M3 = M2
        # for enc in self.model_enc_blks: M3 = enc(M3, cmask)
        p1, p2 = self.out(mod1, mod2, mod3, cmask)
        return p1, p2

class Embedding(nn.Module):
    def __init__(self, word_vectors, char_vocab_size, word_emb_size, char_emb_size, drop_prob_char, drop_prob_word):
        """QAnet embedding Layer
            https://arxiv.org/pdf/1804.09541.pdf
        Args:
            @param word_vectors : Pre-trained word vectors.
            @oaram char_vocab_size : Size of character vocabulary
            @param word_emb_size: Size of character embedding
            @param dropout_prob_char : dropout probability of character embedding 
            @param dropout_prob_word: dropout probability of word embedding
        """
        super(Embedding, self).__init__()

        ######################## PARAMETERS #############################
        self.drop_prob_char = drop_prob_char
        self.drop_prob_word = drop_prob_word

        ##################### Initialize layers #########################
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.char_embed = ModelCharEmbeddings(char_vocab_size,word_emb_size,char_emb_size, drop_prob_char)
        self.hwy = HighwayEncoder(2, word_vectors.size(1) + word_emb_size)

    def forward(self, x, y):
        """
        @param x: Index of words (batch_size, seq_length)
        @param y: Index of characters (batch_size, seq_len, max_word_len)

        @out (batch_size, seq_len, glove_dim + word_emb_size)
        """
        emb_word = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb_char = self.char_embed(y) # (batch_size, seq_len, embed_size)
        emb = torch.cat([F.dropout(emb_word, self.drop_prob_word, self.training),
                         emb_char],dim=2)

        emb = self.hwy(emb)   # (batch_size, seq_len, glove_dim + word_emb_size)
        return emb

class PositionEncoder(nn.Module):
    def __init__(self, hidden_size, max_length = 600):
        """ Postion Encoder from: Attention is all you need
            https://arxiv.org/pdf/1706.03762.pdf
        Args: 
            @param hidden_size: hidden dimension of QAnet model
            @param max_length: maximum length of context 
        """
        super(PositionEncoder, self).__init__()
        #Parameters
        self.hidden_size = hidden_size
        self.max_length = max_length

        #Creating Signal to add
        pos = torch.arange(max_length).float()
        i = torch.arange(self.hidden_size//2)

        sin = torch.ones(self.max_length,self.hidden_size//2).transpose(0,1) * pos
        cos = torch.ones(self.max_length,self.hidden_size//2).transpose(0,1) * pos

        sin = torch.sin(sin.transpose(0,1) / (10000)**(2*i/self.hidden_size))
        cos = torch.cos(cos.transpose(0,1) / (10000)**(2*i/self.hidden_size))

        self.signal2 = torch.zeros((sin.shape[0], 2*sin.shape[1]))
        self.signal2[:,:-1:2] = sin
        self.signal2[:,1::2] = cos
        device, _ = get_available_devices()
        self.signal2 = self.signal2.to(device)

    def forward(self, x):
        return x + self.signal2[:x.shape[1],:]
        

class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])
#        for gate,trans in zip(self.gates, self.transforms): 
            #nn.init.kaiming_normal_(gate.weight, nonlinearity= 'sigmoid')
            #nn.init.kaiming_normal_(trans.weight, nonlinearity= 'relu')

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class CNN(nn.Module):

    def __init__(self,embed_size,char_embed_size=50,ker=5,pad=1):
        """
        Constructor for the gate model
        @param embed_size : int for the size of the word  embeded size
        @param char_embded_size  : int for the size of the caracter  embeded size
        @param ker : int kernel_size used in Convolutions
        @param pad : int padding used in Convolutions
        @param stri : int  number of stride.
        """
        super(CNN, self).__init__() 

        ##################### Initialize layers #########################
        self.conv_layer=nn.Conv1d(in_channels=char_embed_size, out_channels=embed_size, kernel_size=ker, padding=pad)
        #nn.init.kaiming_normal_(self.conv_layer.weight, nonlinearity='relu')
        self.maxpool=nn.AdaptiveMaxPool1d(1)
        
    def forward(self,xreshaped):
        """
        forward function for computing the output
        @param xreshaped : torch tensor of size [BATCH_SIZE, EMBED_SIZE, max_word_lenght]. 
        @return xconvout : torch tensor after convolution and maxpooling [BATCH_SIZE, EMBED_SIZE].
        """
        xconv=self.conv_layer(xreshaped)
        xconvout=self.maxpool(F.relu(xconv)).squeeze()
        return xconvout


class ModelCharEmbeddings(nn.Module): 

    def __init__(self, char_vocab_size, word_embed_size, char_emb_size=50, prob=0.2):
        """QAnet embedding Layer
                https://arxiv.org/pdf/1804.09541.pdf
            Args:
                @param char_vocab_size : Size of character vocabulary
                @param word_emb_size: Size of character embedding
                @param char_emb_size: Size of character embeddings
                @param dropout_prob_char: dropout probability of character embedding 
        """
        super(ModelCharEmbeddings, self).__init__()

        ######################## PARAMETERS #############################
        self.char_vocab_size = char_vocab_size
        self.word_embed_size = word_embed_size
        self.char_emb_size=char_emb_size
        self.prob = prob
        pad_token_idx = 0 

        ##################### Initialize layers #########################
        self.model_embeddings=nn.Embedding(self.char_vocab_size,self.char_emb_size,pad_token_idx)
        self.convnet=CNN(self.word_embed_size,self.char_emb_size)
        

    def forward(self, input):

        batch_size, seq_len, word_len = input.shape
        x_emb = self.model_embeddings(input)
        x_emb = F.dropout(x_emb, self.prob, self.training)

        x_flat = x_emb.flatten(start_dim=0, end_dim = 1)
        x_conv_out = self.convnet(x_flat.permute((0,2,1)))
        return x_conv_out.view((-1,seq_len,self.word_embed_size))
