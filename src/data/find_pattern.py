import re

def find_pattern(log_path):
    data = []

    blk_pattern = re.compile(r'(blk_-?\d+)')

    with open(log_path) as f:
        for line in f:
            blk_match = blk_pattern.search(line)
            if not blk_match:
                continue

            blk_id = blk_match.group(1)

            parts = line.split(':', maxsplit=1)
            if len(parts) < 2:
                continue
            msg = parts[1].strip()

            data.append((blk_id, msg))

    return data