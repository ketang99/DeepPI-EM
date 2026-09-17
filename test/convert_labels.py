from pathlib import Path
import numpy as np
import tifffile
from PIL import Image

src_dir = Path("/home/xgupke/Documents/projects/DeepPI-EM/data/test/target")
dst_dir = src_dir

dst_dir.mkdir(parents=True, exist_ok=True)

for path in sorted(src_dir.glob("*.ome.tif")):

    mask = tifffile.imread(path)
    mask = np.squeeze(mask)

    if mask.ndim != 2:
        raise ValueError(
            f"{path.name}: expected a 2D mask, got shape {mask.shape}"
        )

    # Convert anything > 0 to foreground
    mask = (mask > 0).astype(np.uint8)

    print(
        path.name,
        "shape:", mask.shape,
        "dtype:", mask.dtype,
        "values:", np.unique(mask)
    )

    # Convert 0/1 -> 0/255
    mask_png = mask * 255

    out_name = path.name.replace(".ome.tif", ".png")
    out_path = dst_dir / out_name

    # Save PNG
    Image.fromarray(mask_png).save(out_path)

    print("Saved:", out_path)