from pathlib import Path
import requests
import gzip

def download_data():
    url = 'https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1'
    tgz_path = Path('datasets/HDFS_v1.zip')
    if tgz_path.is_file():
        print('File already present in the right place.')
        return
    else:
        tgz_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with gzip.open(tgz_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print('Download complete')