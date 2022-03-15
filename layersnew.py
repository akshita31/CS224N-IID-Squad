import args
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers_qanet import HighwayEncoder, FeedForwardHelper
from util import masked_softmax

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

    def __init__(self, char_vectors, d_model, drop_char, drop_prob, num_filters) -> None:
        super(_CharEmbedding, self).__init__()
        train_args = args.get_train_args()
        if train_args.use_pretrained_char:
            print('Using the pretrained char embeddings with Freeze = True')
            self.char_embed = nn.Embedding.from_pretrained(char_vectors, freeze = True) #output will be (batch_size, seq_length, chars_per_word, input_embedding_len)
        else:
            print('Using the pretrained char embeddings with Freeze = false')
            self.char_embed = nn.Embedding.from_pretrained(char_vectors, freeze = False) #output will be (batch_size, seq_length, chars_per_word, input_embedding_len)
        self.input_char_emb_size = char_vectors.size(1)
        self.drop_char = drop_char
        self.num_filters = num_filters
        self.drop_prob = drop_prob
        self.d_model = d_model

        self.conv2d = nn.Conv2d(self.input_char_emb_size, d_model, kernel_size=(1,5), padding=0, bias=True)
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')

    def forward(self, char_idxs):

        (batch_size, seq_len, _) = char_idxs.shape

        emb = self.char_embed(char_idxs)
        emb = emb.permute(0, 3, 1, 2)
        emb = F.dropout(emb, self.drop_char, self.training)
        emb = self.conv2d(emb)
        emb = F.relu(emb)
        emb, _ = torch.max(emb, dim=3)
        emb = emb.squeeze()

        assert(emb.shape == (batch_size, self.GetCharEmbedDim(), seq_len))

        return emb

    def GetCharEmbedDim(self):
        return self.d_model

class QANetEmbedding(nn.Module):
   """Combines the Word and Character embedding and then applies a transformation and highway network.
   Output of this layer will be (batch_size, seq_len, hidden_size)
   """

   def __init__(self, word_vectors, char_vectors, d_model, drop_prob, drop_char, num_filters):
       super(QANetEmbedding, self).__init__()
       self.drop_prob = drop_prob
       self.word_embed_size = word_vectors.size(1)
       self.batch_size = word_vectors.size(0)
       self.d_model = d_model

       self.word_embed = nn.Embedding.from_pretrained(word_vectors)
       self.char_embed = _CharEmbedding(char_vectors=char_vectors, drop_prob=drop_prob, d_model = d_model, num_filters = num_filters, drop_char=drop_char)
       self.char_embed_dim = self.char_embed.GetCharEmbedDim()
       self.resizer = FeedForwardHelper(self.word_embed_size + self.char_embed_dim, d_model, bias=False)
       self.hwy = HighwayEncoder(2, d_model)
       print('Using dropout for CharEmbedding as', drop_char)
       print('Using dropout for WordEmbed as', drop_prob)

   def forward(self, word_idxs, char_idxs):
       word_emb = self.word_embed(word_idxs)
       char_emb = self.char_embed(char_idxs)

       (batch_size, seq_len, _) = word_emb.shape
       assert(char_emb.shape == (batch_size, self.char_embed_dim, seq_len))

       word_emb = F.dropout(word_emb, self.drop_prob, self.training)
       word_emb = word_emb.transpose(1,2)
       emb = torch.cat((word_emb, char_emb), dim = 1)
       
       assert(emb.shape == (batch_size, self.word_embed_size + self.char_embed_dim, seq_len))
       emb = self.resizer(emb)
       emb = self.hwy(emb)

       assert(emb.shape == (batch_size, self.GetOutputEmbeddingDim(), seq_len))
       return emb

   def GetOutputEmbeddingDim(self):
       return self.d_model

class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
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
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        print('Using BiDAF attention')
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.empty(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.empty(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.empty(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        
        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, c, q, c_mask, q_mask):

        # c is the context. Its size will be (batch_size, context_len, hidden_size) as this is coming from the hidden state of the RNN
        # q is the query. Its size will be (batch_size, query_len, hidden_size) 

        c = c.transpose(1,2)
        q = q.transpose(1,2)

        batch_size, c_len, hidden_size = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len) #looks like context2query attention as we are setting the places where query has a pad element to 0 through the mask
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q) # a is final context to query attention
        # a denotes the attention vector for each word of the context to the query
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, torch.mul(c,a), torch.mul(c, b)], dim=2)  # (bs, c_len, 4 * hid_size)
        assert(x.shape == (batch_size, c_len, 4 * hidden_size))
        
        return x.transpose(1,2)

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s
