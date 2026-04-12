from pathlib import Path

BASE_DIR = Path().resolve()
LOGFILE_DIR = BASE_DIR / 'datasets' / 'HDFS_v1' / 'HDFS.log'
LABELS_DIR = BASE_DIR / 'datasets' / 'HDFS_v1' / 'anomaly_label.csv'
MODEL_DIR = BASE_DIR / 'checkpoints' / 'final_model.pt'
PREPARED_DF_DIR = BASE_DIR / 'datasets' / 'HDFS_v1' / 'prepared_df.csv'
EVENT_DICT_DIR = BASE_DIR / 'src' / 'utils' / 'event_dict.json'