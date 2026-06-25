import glob
import os
import time

import fsspec

DST = "marin-eu-west4/yael/marin-spin/ising-hot-mix"
fs, _ = fsspec.core.url_to_fs("gs://marin-eu-west4")

files = sorted(glob.glob(os.path.expanduser("~/Downloads/ising-ntp_ising_L16_T*.h5")))
files += sorted(glob.glob("scratch/ising_hot_extra/ising_L16_T*.h5"))
print(f"uploading {len(files)} files -> gs://{DST}/", flush=True)

t0 = time.time()
for i, f in enumerate(files):
    name = os.path.basename(f)
    remote = f"{DST}/{name}"
    if fs.exists(remote) and fs.size(remote) == os.path.getsize(f):
        print(f"  [{i+1}/{len(files)}] skip (exists) {name}", flush=True)
        continue
    fs.put(f, remote)
    print(f"  [{i+1}/{len(files)}] uploaded {name} ({os.path.getsize(f)/1e6:.0f} MB)", flush=True)

print(f"DONE in {time.time()-t0:.0f}s. {len(fs.ls(DST))} files in gs://{DST}/", flush=True)
