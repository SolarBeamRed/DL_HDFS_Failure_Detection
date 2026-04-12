from src.models.model import TunedModel
from src.data.build_training_data import return_loaders
from src.utils.config import MODEL_DIR
from tqdm import trange
import torch
import torch.nn as nn

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, _, vocab_size = return_loaders()

    model = TunedModel(vocab_size).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0058, weight_decay=1.55e-05, betas=(0.81, 0.985))
    epochs = 80
    best_loss = float('inf')
    patience = 10
    counter = 0
    min_delta = 2e-4
    train_losses = []
    val_losses = []

    for epoch in trange(epochs):

        model.train()

        train_loss_epoch = 0
        for X, y in train_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad()

            y_preds = model(X)
            loss = loss_fn(y_preds, y)
            train_loss_epoch += loss.item()
            loss.backward()
            optimizer.step()
        train_losses.append(train_loss_epoch / len(train_loader))

        # Early stopping
        model.eval()
        val_loss_epoch = 0

        for X, y in val_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.no_grad():
                y_preds = model(X)
            loss = loss_fn(y_preds, y)
            val_loss_epoch += loss.item()
        val_loss_epoch /= len(val_loader)
        val_losses.append(val_loss_epoch)

        if val_loss_epoch < best_loss - min_delta:
            best_loss = val_loss_epoch
            best_epoch = epoch
            counter = 0
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_loss': best_loss,
                'counter': counter,
                'vocab_size': vocab_size
            }, MODEL_DIR.parent / 'checkpoint.pth')

        else:
            counter += 1
            if counter >= patience:
                print(
                    f'Early stopping triggered, validation loss has not improved after {patience} rounds.\nBest epoch: {best_epoch}\nBest loss: {best_loss}')
                break

        model.train()

    # Restoring best model
    checkpoint = torch.load(MODEL_DIR.parent / 'checkpoint.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    torch.save(model.state_dict(), MODEL_DIR)
    print(f'Model has been trained. Saved model in {MODEL_DIR.parent}')

if __name__ == "__main__": train_model()