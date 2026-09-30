from pathlib import Path

import numpy as np
from tifffile import imread, imwrite


# Input root
src_root = Path("/mnt/HIVE-EM/Thomas Gerner/exports/Training_Data/rafa-masks")
src_images = src_root / "images"
src_masks = src_root / "masks"

# Output root
dst_root = Path("/home/xgupke/Documents/projects/DeepPI-EM/data_Mut_Ctrl")
dst_norm = dst_root / "input"
dst_labels = dst_root / "target"

dst_norm.mkdir(parents=True, exist_ok=True)
dst_labels.mkdir(parents=True, exist_ok=True)


# There is one OME-TIFF in each folder
image_path = next(src_images.glob("*.ome.tif"))
mask_path = next(src_masks.glob("*.ome.tif"))

image = imread(image_path)
mask = imread(mask_path)

print("Image shape:", image.shape, image.dtype)
print("Mask shape:", mask.shape, mask.dtype)

if image.shape != mask.shape:
    raise ValueError(
        f"Image and mask shapes do not match: "
        f"{image.shape} vs {mask.shape}"
    )

image_name = image_path.name.split('.')[0]
mask_name = mask_path.name.split('.')[0]

print("image_name:", image_name)
print("mask_name: ", mask_name)
print(np.unique(mask))

# for z in range(image.shape[0]):
#     img_plane = image[z]
#     mask_plane = mask[z]

#     # Normalize each 16-bit image plane independently to 0-255
#     img_min = img_plane.min()
#     img_max = img_plane.max()

#     if img_max > img_min:
#         img_norm = (
#             (img_plane.astype(np.float32) - img_min)
#             / (img_max - img_min)
#             * 255
#         ).astype(np.uint8)
#         # invert the image
#         img_norm = 255 - img_norm
#     else:
#         img_norm = np.zeros_like(img_plane, dtype=np.uint8)

#     # Convert mask to binary uint8: 0 or 1
#     mask_uint8 = (mask_plane > 0).astype(np.uint8)

#     # Save each 2D plane separately
#     imwrite(
#         dst_norm / f"{image_name}_{z:02d}.tif",
#         img_norm
#     )

#     imwrite(
#         dst_labels / f"{mask_name}_{z:02d}.tif",
#         mask_uint8
#     )

# print(f"Saved {image.shape[0]} image planes to {dst_norm}")
# print(f"Saved {mask.shape[0]} mask planes to {dst_labels}")