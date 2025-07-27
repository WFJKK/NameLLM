"""
Defines embedding layers, multi-head self-attention, feed-forward networks, transformer blocks,
and a full transformer model for token-based sequence modeling with padding support.
"""

import torch
import torch.nn as nn

from .data import MAX_SEQ_LEN, PADDING_IDX, token_dim


class TokenEmbedding(nn.Module):
    """
    Embedding layer for tokens that maps token indices to dense vectors.

    Args:
        token_dim (int): Number of unique tokens (vocabulary size).
        model_dim (int): Dimension of the embedding vectors.
        padding_idx (int): Index of the padding token to ignore during training.

    Attributes:
        TE (nn.Embedding): Token embedding layer.

    Forward Args:
        x (Tensor): Input tensor of token indices with shape (batch_size, sequence_length).

    Returns:
        Tensor: Embedded tokens with shape (batch_size, sequence_length, model_dim).
    """

    def __init__(self, token_dim, model_dim, padding_idx):
        super().__init__()
        self.TE = nn.Embedding(token_dim + 1, model_dim, padding_idx)

    def forward(self, x):
        output = self.TE(x)
        return output


class PositionEmbedding(nn.Module):
    """
    Positional embedding layer that maps position indices to dense vectors.

    Args:
        model_dim (int): Dimension of the embedding vectors.
        max_seq_len (int): Maximum sequence length.

    Attributes:
        PE (nn.Embedding): Positional embedding layer.

    Forward Args:
        pos (Tensor): Input tensor of position indices with shape (batch_size, sequence_length).

    Returns:
        Tensor: Embedded positions with shape (batch_size, sequence_length, model_dim).
    """

    def __init__(self, model_dim, max_seq_len):
        super().__init__()
        self.PE = nn.Embedding(max_seq_len, model_dim)

    def forward(self, pos):
        output = self.PE(pos)
        return output


class Embedding(nn.Module):
    """
    Combines token and positional embeddings and applies dropout.

    Args:
        model_dim (int): Dimension of embeddings.
        token_dim (int): Number of unique tokens.
        max_seq_len (int): Maximum sequence length.
        padding_idx (int): Index of padding token.
        dropout_value (float, optional): Dropout probability (default=0.1).

    Attributes:
        token_embedding (TokenEmbedding): Token embedding module.
        positional_embedding (PositionEmbedding): Positional embedding module.
        dropout (nn.Dropout): Dropout layer.

    Forward Args:
        x (Tensor): Token indices (batch_size, sequence_length).
        pos (Tensor): Position indices (batch_size, sequence_length).

    Returns:
        Tensor: Combined embeddings with dropout applied
                (batch_size, sequence_length, model_dim).
    """

    def __init__(
        self, model_dim, token_dim, max_seq_len, padding_idx, dropout_value=0.1
    ):
        super().__init__()
        self.token_embedding = TokenEmbedding(token_dim + 1, model_dim, padding_idx)
        self.positional_embedding = PositionEmbedding(model_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x, pos):
        output1 = self.token_embedding(x) + self.positional_embedding(pos)
        output = self.dropout(output1)
        return output


class Multihead(nn.Module):
    """
    Multi-head self-attention mechanism.

    Args:
        model_dim (int): Dimension of input and output embeddings.
        n_heads (int): Number of attention heads.

    Attributes:
        n_heads (int): Number of heads.
        head_dim (int): Dimension per head (model_dim / n_heads).
        W_q, W_k, W_v (nn.Linear): Linear layers to produce queries, keys, values.
        W_o (nn.Linear): Linear layer for output projection.

    Forward Args:
        x (Tensor): Input tensor (batch_size, sequence_length, model_dim).
        padding_mask (Tensor, optional): Boolean mask for padding positions (batch_size, sequence_length).

    Returns:
        Tensor: Output tensor after multi-head attention (batch_size, sequence_length, model_dim).
    """

    def __init__(self, model_dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = model_dim // n_heads
        assert model_dim % n_heads == 0
        self.W_q = nn.Linear(model_dim, model_dim)
        self.W_k = nn.Linear(model_dim, model_dim)
        self.W_v = nn.Linear(model_dim, model_dim)
        self.W_o = nn.Linear(model_dim, model_dim)

    def forward(self, x, padding_mask=None):
        Q = self.W_q(x)  # (B, T, model_dim)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.reshape(Q.shape[0], Q.shape[1], self.n_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # (B, n_heads, T, head_dim)
        K = K.reshape(K.shape[0], K.shape[1], self.n_heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        V = V.reshape(V.shape[0], V.shape[1], self.n_heads, self.head_dim).permute(
            0, 2, 1, 3
        )

        att_score = Q @ K.transpose(-1, -2) / (self.head_dim**0.5)  # (B, n_heads, T, T)

        B, n_heads, T, _ = att_score.shape
        device = x.device
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=device), diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, n_heads, T, T)

        if padding_mask is not None:
            key_padding_mask = (
                padding_mask.unsqueeze(1).unsqueeze(2).expand(B, n_heads, T, T)
            )
            actual_padding_mask = torch.zeros_like(att_score).masked_fill(
                key_padding_mask, float("-inf")
            )
            combined_mask = causal_mask + actual_padding_mask
        else:
            combined_mask = causal_mask

        att_probs = torch.softmax(
            att_score + combined_mask, dim=-1
        )  # (B, n_heads, T, T)
        out = att_probs @ V  # (B, n_heads, T, head_dim)
        out = (
            out.transpose(1, 2).contiguous().reshape(B, T, self.n_heads * self.head_dim)
        )  # (B, T, model_dim)
        out = self.W_o(out)
        return out


class Feedforward(nn.Module):
    """
    Feed-forward network with one hidden layer and ReLU activation.

    Args:
        model_dim (int): Input and output dimension.
        hidden_dim (int): Hidden layer dimension.

    Attributes:
        net (nn.Sequential): Feed-forward network.

    Forward Args:
        x (Tensor): Input tensor (batch_size, sequence_length, model_dim).

    Returns:
        Tensor: Output tensor (batch_size, sequence_length, model_dim).
    """

    def __init__(self, model_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, model_dim),
        )

    def forward(self, x):
        output = self.net(x)
        return output


class Transformerblock(nn.Module):
    """
    Single Transformer block with multi-head attention, feed-forward network, layer normalization,
    dropout, and residual connections.

    Args:
        model_dim (int): Dimension of input/output embeddings.
        num_heads (int): Number of attention heads.
        hidden_dim (int): Dimension of feed-forward hidden layer.
        dropout (float): Dropout probability.

    Attributes:
        MHA (Multihead): Multi-head attention module.
        norm1, norm2 (nn.LayerNorm): Layer normalization layers.
        ffn (Feedforward): Feed-forward network.
        dropout (nn.Dropout): Dropout layer.

    Forward Args:
        x (Tensor): Input tensor (batch_size, sequence_length, model_dim).
        padding_mask (Tensor, optional): Padding mask (batch_size, sequence_length).

    Returns:
        Tensor: Output tensor (batch_size, sequence_length, model_dim).
    """

    def __init__(self, model_dim, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.MHA = Multihead(model_dim, num_heads)
        self.norm1 = nn.LayerNorm(model_dim)
        self.ffn = Feedforward(model_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        x_norm1 = self.norm1(x)
        attn_out = self.MHA(x_norm1, padding_mask=padding_mask)
        attn_out = self.dropout(attn_out)
        x = x + attn_out

        ffn_out = self.ffn(self.norm2(x))
        ffn_out = self.dropout(ffn_out)
        x = x + ffn_out
        return x


class Transformermodel(nn.Module):
    """
    Transformer model combining embeddings, multiple Transformer blocks, and output projection for token prediction.

    Args:
        token_dim (int): Number of unique tokens (vocabulary size).
        model_dim (int): Embedding dimension.
        max_seq_length (int): Maximum input sequence length.
        num_heads (int): Number of attention heads.
        hidden_dim (int): Dimension of feed-forward hidden layers.
        num_layers (int): Number of Transformer blocks.
        padding_idx (int): Padding token index.
        dropout (float): Dropout probability.

    Attributes:
        embedding (Embedding): Combined token and positional embedding module.
        blocks (nn.ModuleList): List of Transformer blocks.
        ln_f (nn.LayerNorm): Final layer normalization.
        head (nn.Linear): Final linear layer projecting to token logits.

    Forward Args:
        token_input (Tensor): Input token indices (batch_size, sequence_length).
        pos_input (Tensor): Input position indices (batch_size, sequence_length).

    Returns:
        Tensor: Logits over vocabulary for each token position (batch_size, sequence_length, token_dim+1).
    """

    def __init__(
        self,
        token_dim,
        model_dim,
        max_seq_length,
        num_heads,
        hidden_dim,
        num_layers,
        padding_idx,
        dropout=0.1,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = Embedding(
            model_dim, token_dim, max_seq_length, padding_idx, dropout
        )
        self.blocks = nn.ModuleList(
            [
                Transformerblock(model_dim, num_heads, hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, token_dim + 1)

    def forward(self, token_input, pos_input):
        padding_mask = token_input == self.padding_idx
        x = self.embedding(token_input, pos_input)
        for block in self.blocks:
            x = block(x, padding_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
