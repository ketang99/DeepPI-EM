"""
Convert ome tif input files to just tifs, because cv2 is being naughty with ometifs.
"""

from pathlib import Path
import tifffile
import numpy as np

src_dir = Path("/home/xgupke/Documents/projects/DeepPI-EM/data/test/input")
dst_dir = src_dir

dst_dir.mkdir(parents=True, exist_ok=True)

for path in sorted(src_dir.glob("*.ome.tif")):
    img = tifffile.imread(path)
    img = np.squeeze(img)

    if img.ndim != 2:
        raise ValueError(
            f"{path.name}: expected 2D image, got shape {img.shape}"
        )

    if img.dtype != np.uint8:
        raise ValueError(
            f"{path.name}: expected uint8, got {img.dtype}"
        )

    out_name = path.name.replace(".ome.tif", ".tif")
    out_path = dst_dir / out_name

    tifffile.imwrite(out_path, img)

    print(
        out_name,
        "shape:", img.shape,
        "dtype:", img.dtype,
        "min/max:", img.min(), img.max()
    )