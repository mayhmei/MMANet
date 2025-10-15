import torch  # Import PyTorch core library
import torch.nn as nn  # Import neural network module
import torch.nn.functional as F  # Import neural network function library
import math  # Import math functions


class MultiHeadAttention(nn.Module):  # Multi-head attention mechanism
    def __init__(self, d_model, num_heads, dropout=0.1):  # Initialization
        super(MultiHeadAttention, self).__init__()  # Call parent initializer
        assert d_model % num_heads == 0  # Ensure model dimension divisible by head count
        self.d_model = d_model  # Model dimension
        self.num_heads = num_heads  # Number of attention heads
        self.d_k = d_model // num_heads  # Per-head dimension
        self.W_q = nn.Linear(d_model, d_model)  # Query linear projection
        self.W_k = nn.Linear(d_model, d_model)  # Key linear projection
        self.W_v = nn.Linear(d_model, d_model)  # Value linear projection
        self.W_o = nn.Linear(d_model, d_model)  # Output linear projection
        self.dropout = nn.Dropout(dropout)  # Dropout to prevent overfitting

    def scaled_dot_product_attention(self, Q, K, V, mask=None):  # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # Compute scaled attention scores
        if mask is not None:  # Apply mask if provided
            scores = scores.masked_fill(mask == 0, -1e9)  # Set masked positions to a very small value
        attention_weights = F.softmax(scores, dim=-1)  # Attention weights
        attention_weights = self.dropout(attention_weights)  # Apply dropout to weights
        output = torch.matmul(attention_weights, V)  # Attention output
        return output

class FeedForward(nn.Module):  # FeedForward neural network
    def __init__(self, d_model, d_ff, dropout=0.1):  # Initialization
        super(FeedForward, self).__init__()  # Call parent initializer
        self.linear1 = nn.Linear(d_model, d_ff)  # First linear layer
        self.linear2 = nn.Linear(d_ff, d_model)  # Second linear layer
        self.dropout = nn.Dropout(dropout)  # Dropout layer

    def forward(self, x):  # Forward pass
        x = self.dropout(F.relu(self.linear1(x)))  # First layer transformation followed by ReLU activation and dropout
        x = self.linear2(x)  # Second layer transformation
        return x

class EncoderLayer(nn.Module):  # Transformer encoder layer
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):  # Initialization
        super(EncoderLayer, self).__init__()  # Call parent initializer
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)  # Self attention layer
        self.feed_forward = FeedForward(d_model, d_ff, dropout)  # FeedForward neural network
        self.norm1 = nn.LayerNorm(d_model)  # First layer normalization
        self.norm2 = nn.LayerNorm(d_model)  # Second layer normalization
        self.dropout = nn.Dropout(dropout)  # Dropout layer

    def forward(self, x, mask=None):  # Forward pass
        attn_output = self.self_attn(x, x, x, mask)  # Compute self attention
        x = self.norm1(x + self.dropout(attn_output))  # Residual connection and layer normalization
        ff_output = self.feed_forward(x)  # FeedForward neural network
        x = self.norm2(x + self.dropout(ff_output))  # Residual connection and layer normalization
        return x

class Model(nn.Module):  # Transformer main model
    def __init__(self, config):  # Initialization
        super(Model, self).__init__()  # Call parent initializer
        self.d_model = config.model.emb  # Model dimension
        self.num_heads = config.model.trf_heads  # Number of attention heads
        self.num_layers = config.model.trf_layers  # Number of encoder layers
        self.d_ff = config.model.trf_feedforward  # Feedforward hidden size
        self.dropout_rate = config.model.trf_dropout  # Dropout rate
        self.max_burst_num = config.model.max_burst_num  # Max number of bursts
        self.use_segment_embedding = getattr(config.model, 'use_segment_embedding', False)  # Use segment embedding
        self.use_position_embedding = getattr(config.model, 'use_position_embedding', True)  # Use positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model)  # Positional encoding layer
        self.segment_embedding = nn.Embedding(self.max_burst_num, self.d_model)  # Segment embedding table
        self.encoder_layers = nn.ModuleList([  # Build multiple encoder layers
            EncoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout_rate)
            for _ in range(self.num_layers)
        ])
        self.dropout = nn.Dropout(self.dropout_rate)  # Dropout layer

    def forward(self, x, burst_indices=None, mask=None):  # Forward pass
        """
        Args:
            x: [batch_size, pad_num, d_model], per-packet representation
            burst_indices: [batch_size, pad_num], burst index per packet
            mask: attention mask
        """
        x = self.pos_encoding(x)  # Add positional encoding
        if self.use_segment_embedding and burst_indices is not None:  # If using segment embedding and burst indices provided
            segment_embeddings = self.segment_embedding(burst_indices)  # Get burst segment embeddings
            x = x + segment_embeddings  # Add segment embeddings to input
        x = self.dropout(x)  # Apply dropout
        for encoder_layer in self.encoder_layers:  # Pass through each encoder layer
            x = encoder_layer(x, mask)  # Encoder layer forward
        return x  # Final output

class PositionalEncoding(nn.Module):  # Positional encoding module
    def __init__(self, d_model, max_len=5000):  # Initialization
        super(PositionalEncoding, self).__init__()  # Call parent initializer
        pe = torch.zeros(max_len, d_model)  # Create position encoding matrix
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Position vector
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # Compute denominator term
        pe[:, 0::2] = torch.sin(position * div_term)  # Compute even positions' sine values
        pe[:, 1::2] = torch.cos(position * div_term)  # Compute odd positions' cosine values
        pe = pe.unsqueeze(0)  # Increase batch dimension
        self.register_buffer('pe', pe)  # Register as buffer, not participate in backward propagation

    def forward(self, x):  # Forward pass
        return x + self.pe[:, :x.size(1)]  # Add positional encoding; select positions matching input length