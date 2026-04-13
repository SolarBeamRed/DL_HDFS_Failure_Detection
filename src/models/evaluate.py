from src.utils.config import MODEL_DIR
from src.data.build_training_data import return_loaders
from src.models.model import TunedModel
import torch
from sklearn.metrics import roc_auc_score, classification_report
from rich.table import Table
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

        model.eval()

        for X, y in test_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.no_grad():
                y_preds = torch.sigmoid(model(X))

                all_preds.extend(y_preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        print()

        binary_preds = [1 if p >= 0.5 else 0 for p in all_preds]

        console.print(f"[green]Final tuned model's ROC-AUC score on test data:[/] {roc_auc_score(all_labels, all_preds)}")

        # Classification report table
        report_dict = classification_report(all_labels, binary_preds, output_dict=True)

        table = Table(title="Classification Report")

        table.add_column("Class")
        table.add_column("Precision")
        table.add_column("Recall")
        table.add_column("F1-score")
        table.add_column("Support")

        for label, metrics in report_dict.items():
            if isinstance(metrics, dict):
                table.add_row(
                    label,
                    f"{metrics['precision']:.2f}",
                    f"{metrics['recall']:.2f}",
                    f"{metrics['f1-score']:.2f}",
                    str(int(metrics['support']))
                )

        console.print(table)

if __name__ == "__main__": evaluate_model()