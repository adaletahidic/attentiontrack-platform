import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TemporalAttention(nn.Module):
    """
    Learns attention weights over timesteps.

    Input:
        h: [B, T, H]
        mask: [B, T] with True for valid timesteps, False for padding

    Output:
        context: [B, H]
        attn_weights: [B, T]
    """
    def __init__(self, hidden_dim: int, attn_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, mask):
        # h -> [B, T, H]
        # energy -> [B, T, A]
        energy = self.tanh(self.proj(h))

        # scores -> [B, T]
        scores = self.score(energy).squeeze(-1)

        # mask padding timesteps so they get zero attention after softmax
        scores = scores.masked_fill(~mask, float("-inf"))

        # attn_weights -> [B, T]
        attn_weights = torch.softmax(scores, dim=1)

        # context -> [B, H]
        context = torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1)
        return context, attn_weights


class AttentionLSTM(nn.Module):
    """
    LSTM classifier with temporal attention pooling over all valid timesteps.
    Compatible with:
        logits = model(X, lengths)
    where
        X: [B, T, F]
        lengths: [B]
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        num_classes: int = 1,
        attn_dim: int = 128,
        fc_hidden: int = 64,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_size * self.num_directions

        self.attention = TemporalAttention(
            hidden_dim=lstm_out_dim,
            attn_dim=attn_dim,
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def make_mask(self, lengths, max_len):
        # mask: [B, T]
        device = lengths.device
        time_ids = torch.arange(max_len, device=device).unsqueeze(0)  # [1, T]
        mask = time_ids < lengths.unsqueeze(1)  # [B, T]
        return mask

    def forward(self, x, lengths, return_attention=False):
        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        max_len = out.size(1)
        mask = self.make_mask(lengths, max_len)

        context, attn_weights = self.attention(out, mask)
        logits = self.classifier(context)

        if return_attention:
            return logits, attn_weights
        return logits