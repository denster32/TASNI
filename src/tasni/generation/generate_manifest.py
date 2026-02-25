import glob
import json
import os
from pathlib import Path

import pandas as pd

OUTPUT_DIR = str(Path(__file__).resolve().parents[3] / "output")
CUTOUT_DIR = os.path.join(OUTPUT_DIR, "cutouts")
MANIFEST_FILE = os.path.join(OUTPUT_DIR, "manifest.json")
CLUSTER_FILE = os.path.join(OUTPUT_DIR, "image_clusters.csv")


def main():
    print("Generating manifest.json for Web UI...")

    # Get files
    files = glob.glob(os.path.join(CUTOUT_DIR, "*.jpg"))
    files = sorted(files)

    # Load clusters if exist
    clusters = {}
    if os.path.exists(CLUSTER_FILE):
        try:
            cdf = pd.read_csv(CLUSTER_FILE)
            # Map filename to cluster
            clusters = dict(zip(cdf["filename"], cdf["cluster"], strict=False))
        except Exception as e:
            print(f"Warning: Could not load clusters: {e}")

    manifest = []
    for f in files:
        if os.path.getsize(f) < 1000:
            continue  # Skip blanks

        basename = os.path.basename(f)
        # Extract ID
        # dr9_J000017.78p132338.0.jpg
        obj_id = basename.replace("dr9_", "").replace(".jpg", "")

        item = {"filename": basename, "id": obj_id, "cluster": int(clusters.get(basename, -1))}
        manifest.append(item)

    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f)

    print(f"Saved {len(manifest)} items to {MANIFEST_FILE}")


if __name__ == "__main__":
    main()
