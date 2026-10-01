from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tifffile import imread
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from model import initialize_model
from config import device, TEST_MODEL_PATH


# -----------------------------
# Paths/settings
# -----------------------------
src_dir = Path("/home/xgupke/Documents/projects/DeepPI-EM/data_Mut_Ctrl/test/input")
labels_dir = src_dir.parent / "target"

labels_dir.mkdir(parents=True, exist_ok=True)

sample_cnt = 4
threshold = 0.25


# -----------------------------
# Load model
# -----------------------------
model = initialize_model()

checkpoint = torch.load(TEST_MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["state_dict"], strict=False)

model.to(device)
model.eval()

print(f"Loaded model from {TEST_MODEL_PATH}")
print(f"Input folder:  {src_dir}")
print(f"Output folder: {labels_dir}")


# -----------------------------
# Run inference
# -----------------------------
ome_files = sorted(src_dir.glob("*.ome.tif"))
if len(ome_files) == 0:
    ome_files = sorted(src_dir.glob("*.tif"))
if len(ome_files) == 0:
    ome_files = sorted(src_dir.glob("*.png"))

with torch.no_grad():
    for src_path in tqdm(ome_files, desc="Predicting"):

        print(f"********\nImage name: {src_path.name}")
        # Load OME-TIFF
        image_np = imread(src_path)
        image_np = np.squeeze(image_np)

        if image_np.ndim != 2:
            raise ValueError(
                f"{src_path.name}: expected a 2D image after squeeze, "
                f"got shape {image_np.shape}"
            )

        # Same basic conversion used by torchvision ToTensor:
        # uint8 [0,255] -> float32 [0,1]
        image = to_tensor(image_np).float()
        image = image.unsqueeze(0).to(device)  # [1, 1, H, W]

        # No user clicks:
        # one dummy positive and one dummy negative point.
        # (-1, -1, -1) is the sentinel used by DeepPI for "no point".
        points = torch.tensor(
            [[
                [-1, -1, -1],
                [-1, -1, -1],
            ]],
            dtype=torch.float32,
            device=device,
        )

        # Forward pass
        output = model(
            image=image,
            points=points,
            sample_cnt=sample_cnt,
            training=False,
        )

        # DeepPI produces multiple probabilistic samples.
        # This follows the predictor logic: sigmoid each sample,
        # then average them.
        samples = [
            torch.sigmoid(sample)
            for sample in output["samples"]
        ]

        prob_mask = torch.stack(samples).mean(dim=0)

        # [1, 1, H, W] -> [H, W]
        prob_mask = prob_mask[0, 0].cpu().numpy()

        # Threshold and save as 0/255 uint8 PNG
        pred_mask = (prob_mask >= threshold).astype(np.uint8) * 255

        # foo.ome.tif -> foo.png
        name = src_path.name
        if name.endswith(".ome.tif"):
            name = name[:-8]

        dst_path = labels_dir / f"{name}.png"

        Image.fromarray(pred_mask).save(dst_path)

        print("Mask predicted")


print(f"\nDone. Predicted masks saved to:\n{labels_dir}")