import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import umap
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from torchvision import models, transforms

# Configuration
CUTOUT_DIR = "./data/processed/cutouts"
OUTPUT_FILE = "./data/processed/image_clusters.csv"
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    # Use efficientnet for feature extraction
    print(f"Loading ResNet50 on {DEVICE}...")
    model = models.resnet50(pretrained=True)
    # Strip the classification layer
    modules = list(model.children())[:-1]
    model = nn.Sequential(*modules)
    model.to(DEVICE)
    model.eval()
    return model


def get_image_files():
    files = glob.glob(os.path.join(CUTOUT_DIR, "*.jpg"))
    # Filter out empty/failed images (small size)
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
            valid_batch_indices = []

            for f in batch_files:
                try:
                    img = Image.open(f).convert("RGB")
                    batch_imgs.append(transform(img))
                    # Extract ID from filename: dr9_J000017.78p132338.0.jpg -> J000017.78+132338.0
                    basename = os.path.basename(f)
                    # Rough ID extraction
                    ids.append(basename)
                except Exception as e:
                    print(f"Error reading {f}: {e}")

            if not batch_imgs:
                continue

            batch_tensor = torch.stack(batch_imgs).to(DEVICE)
            output = model(batch_tensor)
            # Flatten: [Batch, 2048, 1, 1] -> [Batch, 2048]
            output = output.view(output.size(0), -1)
            features.append(output.cpu().numpy())

            if i % 100 == 0:
                print(f"Processed {i}/{len(files)}...", end="\r")

    return np.vstack(features), ids


def cluster_features(features, ids):
    print("\nReducing dimensions (PCA)...")
    pca = PCA(n_components=50)
    pca_features = pca.fit_transform(features)

    print("Reducing dimensions (UMAP) for clustering...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(pca_features)

    print("Clustering (DBSCAN)...")
    # DBSCAN helps find "Noise" (-1) vs Groups
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(embedding)

    df = pd.DataFrame(
        {
            "filename": ids,
            "cluster": clustering.labels_,
            "umap_x": embedding[:, 0],
            "umap_y": embedding[:, 1],
        }
    )

    return df


def main():
    files = get_image_files()
    if len(files) < 50:
        print("Not enough images to cluster yet. Wait for download.")
        return

    model = load_model()
    features, ids = extract_features(model, files)

    print(f"Extracted features shape: {features.shape}")

    df = cluster_features(features, ids)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved clusters to {OUTPUT_FILE}")
    print("Cluster counts:")
    print(df["cluster"].value_counts())


if __name__ == "__main__":
    main()
