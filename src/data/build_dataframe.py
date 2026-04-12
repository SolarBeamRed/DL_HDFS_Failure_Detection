from src.data.build_sequences import build_sequence
from src.data.encode_data import build_event_mapping
from src.data.find_pattern import find_pattern
from src.data.normalize_message import normalise_message
import pandas as pd

def build_dataframe(log_path, training=True):
    parsed_data = find_pattern(log_path)
    parsed_data = [(blk_id, normalise_message(msg)) for blk_id, msg in parsed_data]
    parsed_data = build_event_mapping(parsed_data, training)
    sequences = build_sequence(parsed_data)

    df = pd.DataFrame(sequences.items(), columns=['blk_id', 'events_sequence'])

    return df