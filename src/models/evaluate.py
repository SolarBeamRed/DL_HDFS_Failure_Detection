from sklearn.metrics import roc_auc_score

from src.utils.config import MODEL_DIR
from src.data.build_training_data import return_loaders
from src.models.model import TunedModel
import torch
from rich.console import Console
console = Console()

def evaluate_model():
    if not MODEL_DIR.exists():
        print(f'Model not found in {MODEL_DIR.parent}. Make sure to run training and verify name of .pt file before evaluating model.\n[model should be named "final_model.pt"]')
        return

    with console.status(f"[green][bold]Evaluating...(this could take a few minutes, don't panic!)", spinner_style='arc'):
        _, _ , test_loader, vocab_size = return_loaders()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TunedModel(vocab_size)
        state_dict = torch.load(MODEL_DIR, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)

        all_labels, all_preds = [], []

        for X, y in test_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            model.eval()

            with torch.no_grad():
                y_preds = torch.sigmoid(model(X))

                all_preds.extend(y_preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        print()

        console.print(f"[green]Final tuned model's score on test data:[/] {roc_auc_score(all_labels, all_preds)}")

if __name__ == "__main__": evaluate_model()