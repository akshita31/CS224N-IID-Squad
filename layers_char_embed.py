import torch
import torch.nn as nn
import torch.nn.Functional as F

from layers import HighwayEncoder


class WordPlusCharEmdedding(nn.Module):
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob) -> None:
        super(WordPlusCharEmdedding, self).__init__()
        
        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        
        # to do: initialize the CharCNN correctly here
        self.char_embed = nn.Sequential(nn.Embedding.from_pretrained(char_vectors), # transform char embeddings to the vectors
                                        # pass the embeddings through the CNN
                                        CharCNN(in_channels = char_vectors.size(1), out_channels =  33, kernel_size = 3, stride=2)) # check dimensions passed here

    def forward(self, word_idxs, char_idxs):
        word_embedding = self.word_embed(word_idxs)   # (batch_size, seq_len, embed_size)
        char_embedding = self.char_embed(char_idxs)

        # to do:  Figure out how to concatenate the word and char embedding, maybe something like (need to take care of the dimensions here)
        emb = torch.cat(word_embedding, char_embedding)
        return emb

class CharCNN(nn.Module):
    """CNN which takes in char vectors as input and converts it to word level character embeddings"""
    def __init__(self, in_channels, out_channels, kernel_size, num_filters):
        super(CharCNN, self).__init__()

        # to do : initialize the cnn layers here, maybe use nn.ModuleList or nn.Sequential
        self.cnn = nn.Conv1d(in_channels = in_channels, out_channels =  out_channels, kernel_size = kernel_size)

    def forward(self, x):
        emb = self.cnn(x) # do we need any dropout here ?

        return emb

class CharEmbedding(nn.Module):
    """Character Embedding layer used by BiDAF.

    It takes in an input vector word (or its index) and using the characters in the word, 
    transforms it to an embedding of a fixed size.

    Args:
        char_vector: Pretrained character vectors. (maybe one-hot. need to verify this)
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """

    """ From the paper
Following Kim (2014), we obtain the character level embedding of each word using Convolutional Neural Networks (CNN). Characters are embedded into vectors, 
which can be considered as 1D inputs to the CNN, and whose size is the input channel size of the CNN. The outputs of the CNN are max-pooled over the entire width to obtain a
fixed-size vector for each word.

We use 100 1D filters for CNN char embedding, each with a width of 5.
        """


    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob):
        super(CharEmbedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = WordPlusCharEmdedding(word_vectors=word_vectors, char_vectors=char_vectors, hidden_size=hidden_size, drop_prob=drop_prob)
        
        # 2. Change the input size of the projection layer to be cnn_output_size + word_embed_size. 
        # Projection layer will project it to the hidden_size, after that the whole algo should most likely remain the same.  
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        
        # Projection layer has size (embed_size, hidden_size)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, word_idxs, char_idxs):
        emb = self.embed(word_idxs, char_idxs)
        
        # to do: Check whether the dropout mentioned is needed ?
        # to do: how has the char and word embedding fed into the highway network ??
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb