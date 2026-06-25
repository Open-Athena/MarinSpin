import os
import time

import fsspec

fs, _ = fsspec.core.url_to_fs("gs://marin-eu-west4")
PERM = "marin-eu-west4/grug/marin-spin-v1-a97e72/checkpoints/step-200000"
TEMP = "marin-eu-west4/tmp/ttl=14d/checkpoints-temp/marin-eu-west4/grug/marin-spin-v1-a97e72/checkpoints/step-200000"
PARENT = "scratch/ckpt-aug-200k-final"


def committed(path):
    return fs.exists(path) and fs.exists(f"{path}/manifest.ocdbt") and fs.exists(f"{path}/metadata.json")


for i in range(120):  # up to ~2h at 60s
    src = PERM if committed(PERM) else (TEMP if committed(TEMP) else None)
    if src:
        os.makedirs(PARENT, exist_ok=True)
        fs.get(src, PARENT, recursive=True)  # lands as PARENT/step-200000
        local = f"{PARENT}/step-200000"
        files = sum(len(fl) for _, _, fl in os.walk(local))
        print(f"DOWNLOADED final from {'PERM' if src == PERM else 'TEMP'}: {src}", flush=True)
        print(f"  -> {local} ({files} files)", flush=True)
        break
    time.sleep(60)
else:
    print("TIMEOUT: step-200000 never committed", flush=True)
