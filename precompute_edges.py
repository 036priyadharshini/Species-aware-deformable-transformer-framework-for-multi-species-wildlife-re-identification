import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def npy_path(cache_dir: str, img_relative_path: str) -> Path:
    h = hashlib.md5(img_relative_path.encode()).hexdigest()
    return Path(cache_dir) / f"{h}.npy"


def compute_edge(img_path: Path, image_size: int = 256) -> np.ndarray:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((image_size, image_size), dtype=np.uint8)
    img   = cv2.resize(img, (image_size, image_size))
    edges = cv2.Canny(img, threshold1=50, threshold2=150)
    return edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',  type=str, default='data/wildlifereid-10k')
    parser.add_argument('--cache_dir',  type=str, default='edge_cache')
    parser.add_argument('--image_size', type=int, default=256)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = data_root / 'metadata.csv'
    if not metadata_file.exists():
        raise FileNotFoundError(f"metadata.csv not found at {metadata_file}")

    meta = pd.read_csv(metadata_file, low_memory=False)
    print(f"Total images in metadata: {len(meta)}")

    skipped = 0
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Computing edges"):
        img_path  = data_root / row['path']
        cache_key = str(row['path'])
        out_path  = npy_path(args.cache_dir, cache_key)

        if out_path.exists():
            continue
        if not img_path.exists():
            skipped += 1
            continue

        edge = compute_edge(img_path, args.image_size)
        np.save(out_path, edge)

    print(f"Done. Skipped (missing files): {skipped}")


if __name__ == '__main__':
    main()
