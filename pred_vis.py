import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import LungSegmentationDataset
from models import get_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def minmax_normalize(img):
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img


def get_sample_name(dataset, idx):
    if hasattr(dataset, "samples"):
        image_path, _ = dataset.samples[idx]
        return Path(image_path).stem

    return f"sample_{idx+1}"

def vis_pred(
        img_dir,
        mask_dir,
        model_path,
        model_name,
        output_dir="pred_vis",
        img_size = 256,
        num_examples=5
        ):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    dataset = LungSegmentationDataset(img_dir=img_dir, mask_dir=mask_dir, img_size=img_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = get_model(model_name).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        for idx, (image, mask) in enumerate(loader):
            if idx >= num_examples:
                break

            sample_name = get_sample_name(dataset, idx)

            image = image.to(DEVICE)
            output = model(image)
            pred = torch.sigmoid(output)
            pred = (pred > 0.5).float()

            image_np = image.cpu().squeeze().numpy()
            mask_np = mask.cpu().squeeze().numpy()
            pred_np = pred.cpu().squeeze().numpy()

            # Normalize image for display
            image_np = minmax_normalize(image_np)

            mask_np = (mask_np > 0.5).astype(float)
            pred_np = (pred_np > 0.5).astype(float)

            # Overlay only prediction lung region
            pred_overlay = pred_np.copy()
            pred_overlay[pred_overlay == 0] = float("nan")

            plt.figure(figsize=(14, 4))



            # plt.figure(figsize=(9, 3))

            plt.subplot(1, 4, 1)
            plt.imshow(image_np, cmap="gray")
            plt.title("Input X-ray")
            plt.axis("off")

            # Ground truth solid mask
            plt.subplot(1, 4, 2)
            plt.imshow(mask_np, cmap="gray", vmin=0, vmax=1)
            plt.title("Ground Truth Mask")
            plt.axis("off")

            # Predicted solid mask
            plt.subplot(1, 4, 3)
            plt.imshow(pred_np, cmap="gray", vmin=0, vmax=1)
            plt.title("Predicted Mask")

            plt.axis("off")

            plt.suptitle(f"{model_name.upper()} | {sample_name}", fontsize=14)

            plt.tight_layout()
            save_path = os.path.join(
                output_dir,
                f"prediction_{idx+1:03d}_{sample_name}.png"
            )
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

            plt.close()

            print(f"Saved {save_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["unet", "resunet"])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--output_dir", default="pred_vis")
    parser.add_argument("--num_examples", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    vis_pred(
        img_dir=args.image_dir,
        mask_dir=args.mask_dir,
        model_path=args.model_path,
        model_name=args.model,
        output_dir=args.output_dir,
        num_examples=args.num_examples,
        img_size=args.image_size,
    )
