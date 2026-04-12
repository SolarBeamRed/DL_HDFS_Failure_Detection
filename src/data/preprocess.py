import torch
from torch.nn.utils.rnn import pad_sequence

def process_X(X, max_length=40):
    X = [seq[-max_length:] for seq in X]
    X = pad_sequence(X, batch_first=True, padding_value=0)

    return X

def process_y(y):
    return torch.tensor(y.values, dtype=torch.float32)