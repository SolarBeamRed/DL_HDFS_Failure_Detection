from src.utils.config import MODEL_DIR
from src.models.model import TunedModel
from torchinfo import summary
from rich.console import Console
console = Console()

def summarise_model():
    if not MODEL_DIR.exists():
        console.print(f'Model not found in {MODEL_DIR.parent}. Make sure to run train.py first', style='red bold')
        return

    model = TunedModel()
    summary(model)

if __name__=='__main__': summarise_model()