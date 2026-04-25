import argparse
import torch
from torch.utils.data import DataLoader

from dataset import LungSegmentationDataset
from models import get_model
from utils import diceScore, iouScore

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_model(
        img_dir,
        mask_dir,
        model_path,
        model_name = "unet",
        batch_size = 8,
        img_size = 256
        ):

    dataset = LungSegmentationDataset(
        img_dir=img_dir, 
        mask_dir=mask_dir, 
        img_size=img_size)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=(DEVICE == "cuda")
    )

    model = get_model(model_name).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    total_dice = 0.0
    total_iou = 0.0
    count = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            outputs = model(images)

            total_dice += diceScore(outputs, masks).item()
            total_iou += iouScore(outputs, masks).item()

    avg_dice = total_dice / len(loader)
    avg_iou = total_iou / len(loader)


    print(f"Model: {model_name}")
    print(f"Checkpoint: {model_path}")
    print(f"Test images: {img_dir}")
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average IoU Score: {avg_iou:.4f}")

    return avg_dice, avg_iou

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["unet", "resunet"])
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    evaluate_model(
        img_dir=args.image_dir,
        mask_dir=args.mask_dir,
        model_path=args.model_path,
        model_name=args.model,
        batch_size=args.batch_size,
        img_size=args.image_size
    )