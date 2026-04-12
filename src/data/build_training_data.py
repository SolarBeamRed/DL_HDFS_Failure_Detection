from src.utils.config import LOGFILE_DIR, LABELS_DIR, PREPARED_DF_DIR
from src.data.build_dataframe import build_dataframe
from src.data.preprocess import process_X, process_y
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast

class LogDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def return_df():
    if PREPARED_DF_DIR.exists():
        df = pd.read_csv(PREPARED_DF_DIR)
        df['events_sequence'] = df['events_sequence'].apply(ast.literal_eval)
        return df

    df = build_dataframe(LOGFILE_DIR, training=True)
    labels_df = pd.read_csv(LABELS_DIR)
    labels_df.rename(columns={'BlockId': 'blk_id'}, inplace=True)

    df = df.merge(labels_df, on='blk_id')
    df['Label'] = df['Label'].map({'Normal': 0, 'Anomaly': 1})

    return df

def return_loaders():
    df = return_df()

    X = df['events_sequence']
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2,
                                                      random_state=42)
    max_length = 40

    X_train = process_X(X_train, max_length)
    X_val = process_X(X_val, max_length)
    X_test = process_X(X_test, max_length)

    y_train = process_y(y_train)
    y_val = process_y(y_val)
    y_test = process_y(y_test)

    vocab_size = max(X_train.max(), X_test.max()) + 1

    train = LogDataset(X_train, y_train)
    val = LogDataset(X_val, y_val)
    test = LogDataset(X_test, y_test)

    train_loader = DataLoader(train, batch_size=128, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=3)
    val_loader = DataLoader(val, batch_size=128, shuffle=False, pin_memory=True, num_workers=8, prefetch_factor=3)
    test_loader = DataLoader(test, batch_size=128, shuffle=False, pin_memory=True, num_workers=8, prefetch_factor=3)

    return train_loader, val_loader, test_loader, vocab_size