from src.utils.config import EVENT_DICT_DIR
import json

def build_event_mapping(normalised_data, training=True):
    encoded_data = []
    if EVENT_DICT_DIR.exists():
        with open(EVENT_DICT_DIR) as f:
            event_dict = json.load(f)
        event_id = max(event_dict.values(), default=1) + 1
    else:
        event_dict = {}
        event_id = 2 # 0 for padding, 1 for UNKNOWN, hence starting from 2

    for blk_id, msg in normalised_data:
        if msg not in event_dict:
            if training:
                event_dict[msg] = event_id
                event_id += 1
            else:
                encoded_data.append((blk_id, 1))  # 1=UNKNOWN
                continue

        encoded_data.append((blk_id, event_dict[msg]))

    if training:
        with open(EVENT_DICT_DIR, 'w') as f:
            json.dump(event_dict, f)
    return encoded_data