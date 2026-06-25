import glob
import os
import time

import fsspec

fs, _ = fsspec.core.url_to_fs("gs://marin-eu-west4")
SRC = "marin-eu-west4/yael/marin-spin/ising-hot-mix"        # 14 originals + 8 hot-extra (already on GCS)
DST = "marin-eu-west4/yael/marin-spin/ising-enriched"

# 1) server-side copy the 22 existing files (in-region, fast)
existing = fs.ls(SRC)
for p in existing:
    fs.copy(p, f"{DST}/{p.rsplit('/', 1)[1]}")
print(f"copied {len(existing)} existing files -> {DST}", flush=True)

# 2) upload the 6 cold-extra (local-only)
cold = sorted(glob.glob("scratch/ising_cold_extra/ising_L16_T*.h5"))
t0 = time.time()
for i, f in enumerate(cold):
    name = os.path.basename(f)
    fs.put(f, f"{DST}/{name}")
    print(f"  [{i+1}/{len(cold)}] uploaded {name} ({os.path.getsize(f)/1e6:.0f} MB)", flush=True)

n = len(fs.ls(DST))
print(f"DONE in {time.time()-t0:.0f}s. ising-enriched has {n} files (expect 28)", flush=True)
