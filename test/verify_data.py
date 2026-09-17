# from pathlib import Path
# import numpy as np
# import tifffile

# img_dir = Path("/home/xgupke/Documents/projects/DeepPI-EM/data/train/input")

# for path in sorted(img_dir.glob("*.tif")):

#     print("\n", path.name)

#     try:
#         img = tifffile.imread(path)
#     except Exception as e:
#         print("FAILED TO READ:", e)
#         continue

#     img = np.squeeze(img)

#     print("shape:", img.shape)
#     print("dtype:", img.dtype)
#     print("min/max:", img.min(), img.max())

#     if img.ndim != 2:
#         print("WARNING: not a single-channel 2D image")

#     if img.dtype != np.uint8:
#         print("WARNING: expected uint8")

from pathlib import Path
import cv2
import numpy as np

img_dir = Path("/home/xgupke/Documents/projects/DeepPI-EM/data/train/target")
print(list(img_dir.glob("*.tif")))

for path in sorted(img_dir.glob("*.png")):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    print("\n", path.name)

    if img is None:
        print("FAILED TO READ")
        continue

    print("shape:", img.shape)
    print("dtype:", img.dtype)
    print("min/max:", img.min(), img.max())
    print("unique values: ", np.unique(img))

    if img.ndim != 2:
        print("WARNING: not a single-channel 2D image")

    if img.dtype != np.uint8:
        print("WARNING: expected uint8")