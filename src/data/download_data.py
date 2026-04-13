from src.utils.config import LOGFILE_DIR, LABELS_DIR
import requests
import zipfile
import shutil
from pathlib import Path
from rich.console import Console
from tqdm import tqdm
console = Console()

def download_data():
    if LOGFILE_DIR.exists() and LABELS_DIR.exists():
        return
    elif LOGFILE_DIR.exists() or LABELS_DIR.exists():
        console.print("Partial dataset found. Cleaning and Re-downloading...", style="yellow")
        shutil.rmtree(LOGFILE_DIR.parent, ignore_errors=True)

    url = 'https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1'
    zip_path = Path('datasets/HDFS_v1.zip')
    extraction_dir = Path('datasets')

    extraction_dir.mkdir(parents=True, exist_ok=True)
    console.print(f'[green]Downloading Data...')
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total = int(response.headers.get('content-length', 0))
    with open(zip_path,'wb') as f:
        for chunk in tqdm(response.iter_content(chunk_size=8192), total=total//8192):
            if chunk:
                f.write(chunk)
    console.print('Download complete!', style='green')

    with console.status(f'[green]Extracting data...', spinner_style='arc'):
        with zipfile.ZipFile(zip_path, 'r') as f:
            f.extractall(extraction_dir)

    console.print('Extraction complete!', style='green')

    if not LOGFILE_DIR.exists() or not LABELS_DIR.exists():
        raise FileNotFoundError(
            "Dataset downloaded but expected files not found. Check extraction structure."
        )