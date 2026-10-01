from pathlib import Path
from PIL import Image
import numpy as np

mask_dir = Path("/home/xgupke/Documents/projects/DeepPI-EM/data_Mut_Ctrl/test/target")

for tif_path in mask_dir.glob("*.tif"):
    with Image.open(tif_path) as img:
        mask = np.array(img)

    mask = (mask * 255).astype(np.uint8)

    png_path = tif_path.with_suffix(".png")
    Image.fromarray(mask).save(png_path)

    tif_path.unlink()

    print(f"{tif_path.name} -> {png_path.name}, deleted original TIFF")