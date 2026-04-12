from src.data.build_dataframe import build_dataframe
from src.data.preprocess import process_X
from src.models.model import TunedModel
from src.utils.config import MODEL_DIR
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from rich.console import Console
console = Console()
import torch

class InferenceDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

def predict():
    if not MODEL_DIR.exists():
        console.print(f'Model not found in {MODEL_DIR.parent}. Make sure to run train.py first', style='red bold')
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(MODEL_DIR.parent / 'checkpoint.pth', map_location=device)
    vocab_size = checkpoint['vocab_size']
    model = TunedModel(vocab_size)
    state_dict = torch.load(MODEL_DIR, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)


    console.print(f'Enter path to log file: ', end='', style='yellow')
    userfile_path = Path(input())

    if not userfile_path.exists():
        console.print(f'Given path does not exist. Check again!', style='red bold')
        return None
    if not userfile_path.is_file():
        console.print(f'{userfile_path.name} is not a file. Check again!', style='red bold')
        return None
    if not userfile_path.suffix=='.log':
        console.print(f'Given file is not a .log file', style='red bold')
        return None

    with console.status(f'[yellow][italic]Performing inference...(this may take a few minutes)', spinner_style='arc'):
        df = build_dataframe(userfile_path, training=False)
        X = df['events_sequence']
        X = process_X(X, 40)

        dataset = InferenceDataset(X)

        loader = DataLoader(
            dataset,
            batch_size=512,  # tune this
            shuffle=False,
            pin_memory=True
        )

        model.eval()
        preds = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                out = torch.sigmoid(model(batch))
                preds.append(out.cpu())

        probs = torch.cat(preds).numpy()
        df['anomaly_score'] = probs

        return df

def detect_anomalies():
    df = predict()

    if df is None:
        console.print(f'Empty df error, quitting...', style='bold red')
        return None
    df = df.sort_values(by='anomaly_score', ascending=False)
    threshold = 0.5
    df['prediction'] = (df['anomaly_score'] >= threshold).astype(int)
    anomalies = df[df['prediction'] == 1]
    anomalies = anomalies.sort_values(by='anomaly_score', ascending=False)
    console.print('Predictions obtained', style='green')

    total_blocks = len(df)
    anomalous_blocks = len(anomalies)
    normal_blocks = total_blocks - anomalous_blocks
    anomaly_rate = anomalous_blocks / total_blocks if total_blocks > 0 else 0

    # Printing summary
    ###############################################
    print()
    console.rule(characters='=', style='cyan')
    console.print('SUMMARY OF UPLOADED LOG', style='green', justify='center')
    console.rule(characters='=', style='cyan')
    console.print(f'Total blocks: {total_blocks}', style='yellow')
    console.print(f'Normal blocks: {normal_blocks}', style='green')
    console.print(f'Anomalous blocks: {anomalous_blocks}', style='red')
    console.print(f'Anomaly rate: {anomaly_rate:.6f}', style='yellow')
    console.rule(characters='-', style='cyan')
    ###############################################

    while True:
        console.print('What do you want to do?\n1.Save predicted anomalies as csv\n2. See predicted anomaly blocks\n3. Save all predictions into a csv file\n4. Quit\nEnter 1, 2, 3 or 4: ', style=' yellow', end='')
        ch = int(input().strip())

        if ch==1:
            if anomalous_blocks == 0:
                print('No anomalous blocks to save!')
            else:
                console.print(f'Enter name of csv file to be saved (including .csv in the name): ', end='')
                csv_name = input()
                anomalies.to_csv(csv_name, index=False)
                console.print(f'File saved as {csv_name}', style='green bold')

        elif ch == 2:
            if len(anomalies) == 0:
                console.print('No anomalies detected', style='green bold')
            else:
                console.print(f'Detected {len(anomalies)} anomalous blocks. Display all anomalous blocks? (y/n)')
                all_ch = input().strip().lower()
                if all_ch == 'y':
                    console.print(anomalies[['blk_id', 'anomaly_score']])
                else:
                    console.print(f'How many to display? ', style='blue', end='')
                    n_display_blocks = min(int(input()), len(anomalies))
                    console.print(anomalies[['blk_id', 'anomaly_score']].head(n_display_blocks))

        elif ch == 3:
            console.print(f'Enter name of csv file to be saved (including .csv in the name): ', end='')
            csv_name = input()
            df[['blk_id', 'anomaly_score', 'prediction']].to_csv(csv_name, index=False)
            console.print(f'File saved as {csv_name}', style='green bold')

        elif ch == 4:
            console.print('Exiting user prediction operations... Going back to main menu', style='blue')
            break

        else:
            console.print('Please enter a valid input!', style='#FFAC1C')
        console.print('Continue? (y/n)', style='#FFAC1C bold')
        temp_input = input().strip().lower()
        if temp_input != 'y':
            break

    return None