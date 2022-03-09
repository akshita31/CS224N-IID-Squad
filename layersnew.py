import torch
import torch.nn as nn
import torch.nn.functional as F
from layers_qanet import Highway, Initialized_Conv1d

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

   def __init__(self, word_vectors, char_vectors, d_model, drop_prob, num_filters):
       super(QANetEmbedding, self).__init__()
       self.drop_prob = drop_prob
       self.word_embed_size = word_vectors.size(1)
       self.batch_size = word_vectors.size(0)
       self.d_model = d_model

       self.word_embed = nn.Embedding.from_pretrained(word_vectors)
       self.char_embed = _CharEmbedding(char_vectors=char_vectors, drop_prob=drop_prob, num_filters = num_filters)
       self.char_embed_dim = self.char_embed.GetCharEmbedDim()
       self.resizer = Initialized_Conv1d(self.word_embed_size + self.char_embed_dim, d_model, bias=False)
       self.hwy = Highway(2, d_model)

   def forward(self, word_idxs, char_idxs):
       word_emb = self.word_embed(word_idxs)
       char_emb = self.char_embed(char_idxs)

       (batch_size, seq_len, _) = word_emb.shape
       assert(char_emb.shape == (batch_size, seq_len, self.char_embed_dim))

       word_emb = F.dropout(word_emb, self.drop_prob, self.training)

       emb = torch.cat((word_emb, char_emb), dim = 2)
       emb = emb.transpose(1, 2)
       emb = self.resizer(emb)
       emb = self.hwy(emb)

       assert(emb.shape == (batch_size, self.GetOutputEmbeddingDim(), seq_len))
       return emb

   def GetOutputEmbeddingDim(self):
       return self.d_model