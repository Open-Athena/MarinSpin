import time

import fsspec

fs, _ = fsspec.core.url_to_fs("gs://marin-eu-west4")
TRAIN = "marin-eu-west4/yael/marin-spin/L16-splits-hot/train"
VAL = "marin-eu-west4/yael/marin-spin/L16-splits-hot/val"


def count(path):
    try:
        return len([p for p in fs.glob(f"{path}/**/*.jsonl.gz")])
    except Exception:
        return 0


prev_t = -1
stable = 0
for i in range(80):  # up to ~2h at 90s
    t, v = count(TRAIN), count(VAL)
    if t > 0 and t == prev_t:
        stable += 1
    else:
        stable = 0
    prev_t = t
    # done when train+val shards present and train count stable for 3 checks (write finished)
    if t > 0 and v > 0 and stable >= 3:
        print(f"TOKENIZE SHARDS READY: train={t} shards, val={v} shards in L16-splits-hot", flush=True)
        break
    time.sleep(90)
else:
    print(f"TIMEOUT: train={count(TRAIN)} val={count(VAL)} (not stable)", flush=True)
