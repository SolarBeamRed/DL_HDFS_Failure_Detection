from collections import defaultdict

def build_sequence(encoded_data):
    sequences = defaultdict(list)

    for blk, event_id in encoded_data:
        sequences[blk].append(event_id)

    return sequences