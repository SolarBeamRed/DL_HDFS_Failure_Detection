import torch
from torch.nn.utils.rnn import pad_sequence

def process_X(X, max_length=40):
    X = [torch.tensor(seq[-max_length:], dtype=torch.long) for seq in X]
    X = pad_sequence(X, batch_first=True, padding_value=0)

    return X

def process_y(y):
    return torch.tensor(y.values, dtype=torch.float32)