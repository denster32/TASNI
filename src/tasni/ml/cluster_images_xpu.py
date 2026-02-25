import glob
import os

import numpy as np
import pandas as pd
import torch

try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass  # Might be built-in or not needed for basic XPU depending on version
import torch.nn as nn
import umap
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from torchvision import models, transforms

# Configuration
CUTOUT_DIR = "./data/processed/cutouts"
OUTPUT_FILE = "./data/processed/image_clusters_xpu.csv"
BATCH_SIZE = 64
DEVICE = "xpu"


def load_model():
    print(f"Loading ResNet50 on {DEVICE}...")
    model = models.resnet50(pretrained=True)
    modules = list(model.children())[:-1]
    model = nn.Sequential(*modules)
    model.to(DEVICE)
    if "ipex" in globals():
        model = ipex.optimize(model, dtype=torch.float32)
    model.eval()
    return model


def get_image_files():
    files = glob.glob(os.path.join(CUTOUT_DIR, "*.jpg"))
    valid_files = [f for f in files if os.path.getsize(f) > 1000]
    print(f"Found {len(valid_files)} valid images.")
    return valid_files


def extract_features(model, files):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    features = []
    ids = []

    with torch.no_grad():
        for i in range(0, len(files), BATCH_SIZE):
            batch_files = files[i : i + BATCH_SIZE]
            batch_imgs = []

            for f in batch_files:
                try:
                    img = Image.open(f).convert("RGB")
                    batch_imgs.append(transform(img))
                    ids.append(os.path.basename(f))
                except (OSError, ValueError):
                    continue  # Skip corrupted or unreadable images

            if not batch_imgs:
                continue

            batch_tensor = torch.stack(batch_imgs).to(DEVICE)
            output = model(batch_tensor)
            output = output.view(output.size(0), -1)
            features.append(output.cpu().numpy())

            if i % 100 == 0:
                print(f"Processed {i}/{len(files)}...", end="\r")

    return np.vstack(features), ids


def cluster_features(features, ids):
    print("\nClustering (XPU View)...")
    # We can use different parameters here to get a "Second Opinion"
    pca = PCA(n_components=50)
    pca_features = pca.fit_transform(features)

    reducer = umap.UMAP(
        n_neighbors=30, min_dist=0.0, n_components=2, random_state=99
    )  # Tighter clusters
    embedding = reducer.fit_transform(pca_features)

    clustering = DBSCAN(eps=0.3, min_samples=3).fit(embedding)  # Stricter

    df = pd.DataFrame(
        {
            "filename": ids,
            "cluster_xpu": clustering.labels_,
            "xpu_umap_x": embedding[:, 0],
            "xpu_umap_y": embedding[:, 1],
        }
    )

    return df


def main():
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        print("XPU NOT AVAILABLE. Aborting.")
        # Fallback to CPU? No, we want to test XPU.
        return

    files = get_image_files()
    model = load_model()
    features, ids = extract_features(model, files)

    df = cluster_features(features, ids)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved XPU clusters to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
