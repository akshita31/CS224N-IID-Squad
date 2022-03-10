import args
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
       self.resizer = Initialized_Conv1d(self.word_embed_size + self.char_embed_dim, d_model, bias=False)
       self.hwy = Highway(2, d_model)
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