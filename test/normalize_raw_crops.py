from pathlib import Path

import numpy as np
from tifffile import imread, imwrite

src_dir = Path("/home/xgupke/Documents/projects/DeepPI-EM/data_Mut1_Section6_Align/raw")
dst_dir = Path("/home/xgupke/Documents/projects/DeepPI-EM/data_Mut1_Section6_Align/norm")

dst_dir.mkdir(parents=True, exist_ok=True)

for src_path in sorted(src_dir.glob("*.ome.tif")):
    # Load image as numpy array
    img = imread(src_path)

    print(f"{src_path.name}: shape={img.shape}, dtype={img.dtype}")

    # Min-max normalize to [0, 255]
    img = img.astype(np.float32)

    img_min = img.min()
    img_max = img.max()

    if img_max > img_min:
        img_norm = (img - img_min) / (img_max - img_min)
        img_norm = (img_norm * 255).round().astype(np.uint8)
    else:
        img_norm = np.zeros_like(img, dtype=np.uint8)
        print("Invalid intensity range")

    # Save normalized image
    dst_path = dst_dir / src_path.name
    imwrite(dst_path, img_norm)

    print(
        f"  -> saved {dst_path.name}: "
        f"shape={img_norm.shape}, dtype={img_norm.dtype}, "
        f"range=({img_norm.min()}, {img_norm.max()})"
    )