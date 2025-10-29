import torch
import torch.nn as nn
import math

# Represent vocabulary words as vectors
class InputEmbeddings(nn.Module): 
    
    # Constructor
    def __init__(self, d_model: int, vocab_size: int):
        # Call parent class (nn.Module) constructor
        super().__init__()
        
        # Initialize data:
        # - d_model: vector dimensions
        # - vocab_size: number of vectors (vocabulary words)
        # - embedding: dictionary of vocab_size embedding vectors of size d_model
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    # Forward pass: multiply the embedding of the parameter x by the square root of d_model
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) 
    
# Encode the position of each word in the list by adding values to each element in the embedding vectors
class PositionalEncoding(nn.Module):
    
    # Constructor
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        # Call parent class (nn.Module) constructor
        super().__init__()
        
        # Initialize data:
        # - d_model: vector dimensions (same as in InputEmbeddings)
        # - seq_len: maximum length of the sentence
        # - dropout: randomly zero some of the elements of probability "dropout" 
        #            in the input tensor during training to prevent over-fitting
        
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # (seq_len * d_model) matrix encoding positional values for a word
        pe = torch.zeros(seq_len, d_model)
        
        # (seq_len * 1) vector containing possible positions from 0 to seq_len - 1
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        # Populate each element of pe using this formula, the result depending on the position of the corresponding word
        # - use sin for even rows, cos for odd rows
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        self.register_buffer('pe', pe) # save tensor and state
    
    # Forward pass: add the positional encoding values to the parameter x and apply dropout. Positional encoding values
    #  do not change during training.
    def forward(self, x):
        x = x + (self.pe[:, :x.size(1), :]).requires_grad_(False) # type: ignore
        return self.dropout(x)

# Normalize layer values
class LayerNormalization(nn.Module):
    
    # Constructor
    def __init__(self, eps: float = 10**-6) -> None:
        # Call parent class (nn.Module) constructor
        super().__init__()
        
        # Initialize data:
        # - eps: small number added to the denominator of the normalization formula to ensue it never becomes 0
        # - alpha: normalization parameter to be multiplied
        # - bias: normalization parameter to be added
        
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
    
    # Forward pass: apply normalization formula using the mean and standard deviation of x
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

# Feed data forward using two passes of linear transformations
class FeedForwardBlock(nn.Module):
    
    # Constructor
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        # Call parent class (nn.Module) constructor
        super().__init__()
        
        # Initialize data:
        # - linear_1: linear transformation matrix of shape (d_model, d_ff)
        # - linear_2: linear transformation matrix of shape (d_ff, d_model)
        # - dropout: randomly zero some of the elements of probability "dropout" 
        #            in the input tensor during training to prevent over-fitting
        
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    # Forward pass: linearly transform the incoming x twice
    # - (Batch, seq_len, d_model) ==> (Batch, seq_len, d_ff) ==> (Batch, seq_len, d_model)
    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
# Another form of transforming data
class MultiHeadAttentionBlock(nn.Module):
    
    # Constructor
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        # Call parent class (nn.Module) constructor
        super().__init__()
        
        # Initialize data:
        # - d_model: vector dimensions
        # - h: number of "heads" into which to divide the layer
        # - dropout: randomly zero some of the elements of probability "dropout" 
        #            in the input tensor during training to prevent over-fitting
        # - d_k: dimensions of heads
        # - w_q, w_k, w_v: 3 linear transformation matrices of shape (d_model, d_model) 
        #                  to transform 3 input matrices to encode relationship between word pairs
        # - w_o: linear transformation matrix of shape (d_model, d_model) to transform 
        #        matrix derived from the three transformed copies of input data
        
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)
        
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.w_o = nn.Linear(d_model, d_model)
    
    @staticmethod # Apply formula to heads and take the softmax
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        
        # (Batch, h, seq_len, d_k) ==> (Batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # replace matrix elements with a very small value if that word pair is not meant to interact 
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)
        
        # (Batch, h, seq_len, seq_len)
        attention_scores = attention_scores.softmax(dim = -1)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores
    
    # Forward pass: transform 3 copies of traditional input matrix x, divide the results into h heads by column,
    #  take the softmax of corresponding heads, and transform the resulting matrix of the same dimensions as x
    def forward(self, q, k, v, mask):
        # resulting matrices have the same dimensions as the inputs
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        
        # (Batch, seq_len, d_model) ==> (Batch, seq_len, h, d_k) ==> (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # (Batch, h, seq_len, d_k) ==> (Batch, seq_len, h, d_k) ==> (Batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        # x retains its dimensions
        return self.w_o(x)
    
# Add new layer to previous layer and normalize
class ResidualConnection(nn.Module):
    
    # Constructor
    def __init__(self, dropout: float) -> None:
        # Call parent class (nn.Module) constructor
        super().__init__()
        
        # Initialize data:
        # - dropout: randomly zero some of the elements of probability "dropout" 
        #            in the input tensor during training to prevent over-fitting
        
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
# Pass incoming data x through a series of layers to encode it (alt., pass incoming layer x through a series of functions)
class EncoderBlock(nn.Module):
    
    # Constructor
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        # Call parent class (nn.Module) constructor
        super().__init__()
        
        # Initialize data:
        # - self_attention_block: applies multi head attention to x, where x is the query, key, and value 
        #                         (every word in the sentence is related to every other word)
        # - feed_forward_block: applies feed forward to the data
        # - residual_connections: adds and normalizes the data
        
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    # Forward pass: set x to the addition and normalization of itself and the result after applying 
    #  multi head attention, then again after applying feed forward
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
# Pass incoming data x through a number of encoder block sequences
class Encoder(nn.Module):
    # Constructor
    def __init__(self, layers: nn.ModuleList) -> None:
        # Call parent class (nn.Module) constructor
        super().__init__()
        
        # Initialize data:
        # - layers: functions to transform values in x
        # - norm: function to normalize values in x
        
        self.layers = layers
        self.norm = LayerNormalization()
    
    # Forward pass: repeat encoder block sequence n times, the output of one layer being the input of the next layer
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)

# Pass incoming data x through a series of layers to decode it (alt., pass incoming layer x through a series of functions)
class DecoderBlock(nn.ModuleList):
    
    # Constructor
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        # Call parent class (nn.Module) constructor
        super().__init__()
        
        # Initialize data:
        # - self_attention_block: applies multi head attention to x, where x is the query, key, and value 
        #                         (every word in the sentence is related to every other word)
        # - cross_attention_block: applies multi head attention to x, where x is the query, while key and value 
        #                          come from the encoder (determine relationship between input and output words)
        # - feed_forward_block: applies feed forward to the data
        # - residual_connections: adds and normalizes the data
        
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    # Forward pass: set x to the addition and normalization of itself and the result after applying 
    #  self attention, then again after applying cross attention, then again after applying feed forward
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, lambda x: self.feed_forward_block(x))
        return x

# Pass incoming data x through a number of decoder block sequences
class Decoder(nn.Module):
    # Constructor
    def __init__(self, layers: nn.ModuleList) -> None:
        # Call parent class (nn.Module) constructor
        super().__init__()
        
        # Initialize data:
        # - layers: functions to transform values in x
        # - norm: function to normalize values in x
        
        self.layers = layers
        self.norm = LayerNormalization()
    
    # Forward pass: repeat decoder block sequence n times, the output of one layer being used as input for the next layer
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return self.norm(x)

# Convert embedding to a vocabulary position
class ProjectionLayer(nn.Module):
    
    # Constructor
    def __init__(self, d_model: int, vocab_size: int) -> None:
        # Call parent class (nn.Module) constructor
        super().__init__()
        
        # Initialize data:
        # - proj: linear transformation matrix of shape (d_model, vocab_size)
        
        self.proj = nn.Linear(d_model, vocab_size)
    
    # Forward pass: linearly transform the incoming x and take softmax
    # - (Batch, seq_len, d_model) ==> (Batch, seq_len, vocab_size)
    def forward(self, x):
        # (Batch, seq_len, d_model) ==> (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)
    
# Connects encoding, decoding, and projection layers
class Transformer(nn.Module):
    
    # Constructor
    def __init__(
        self, 
        encoder: Encoder, 
        decoder: Decoder, 
        src_embed: InputEmbeddings, 
        tgt_embed: InputEmbeddings, 
        src_pos: PositionalEncoding, 
        tgt_pos: PositionalEncoding, 
        projection_layer: ProjectionLayer
    ) -> None:
        # Call parent class (nn.Module) constructor
        super().__init__()
        
        # Initialize data:
        # - encoder: encode embeddings
        # - decoder: decode embeddings
        # - src_embed, tgt_embed: input and output embeddings respectively
        # - src_pos, tgt_pos: input and output positional encodings respectively
        # - projection_layer: convert embedded output of decoder to a vocabulary position
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    
    # Encode input embeddings
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    # Decode output embeddings using the encoded data
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    # Project decoder output
    def project(self, x):
        return self.projection_layer(x)
    
def build_transformer(
    src_vocab_size: int, 
    tgt_vocab_size: int, 
    src_seq_len: int, 
    tgt_seq_len: int, 
    d_model: int = 512, 
    N: int = 6, 
    h: int = 8, 
    dropout: float = 0.1, 
    d_ff: int = 2048
) -> Transformer:
    
    # Create embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    
    # Create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create encoder blocks to be fed to the encoder
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    # Create decoder blocks to be fed to the decoder
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
        
    # Create encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    # Create projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer