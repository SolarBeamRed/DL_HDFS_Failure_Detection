import re

def normalise_message(msg):
    msg = re.sub(r"blk_-?\d+", "<*>", msg) # Removing blk_id
    msg = re.sub(r"/?\d+\.\d+\.\d+\.\d+(?::\d+)?:?", "<*>", msg) # Removing IP addresses and ports
    msg = re.sub(r"\b\d+\b", "<*>", msg) # Removing standalone numbers like size
    msg = re.sub(r"/[^\s]+", "<*>", msg) # Removing paths

    return msg