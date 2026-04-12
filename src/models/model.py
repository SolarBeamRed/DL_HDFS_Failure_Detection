import torch.nn as nn
import torch

class TunedModel(nn.Module):
    def __init__(self, vocab_size=55):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=128, padding_idx=0)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True, dropout=0.3,
                            bidirectional=True)
        self.fc = nn.Linear(64 * 2, 1)

    def forward(self, x):
        x = self.embed(x)
        _, (h_n, _) = self.lstm(x)

        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_last = torch.cat((h_forward, h_backward), dim=1)

        out = self.fc(h_last)
        return out.squeeze(1)