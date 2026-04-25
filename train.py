import argparse
import torch
from torch.utils.data import DataLoader, random_split

from dataset import LungSegmentationDataset
from models import get_model
from utils import diceScore, iouScore, DiceBCELoss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True


def train(
        img_dir,
        mask_dir,
        model_name = "unet",
        dataset_name = "SHCXR",
        epochs = 25,
        batch_size = 8,
        lr = 1e-4,
        img_size = 256,
        val_split=0.2,
        seed=42
        ):
    print(f"Using device: {DEVICE}")
    print(f"Training {model_name} on {dataset_name}")

    dataset = LungSegmentationDataset(
        img_dir=img_dir, 
        mask_dir=mask_dir, 
        img_size=img_size)
    
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        dataset, 
        [train_size, val_size], 
        generator=generator
    )

    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=(DEVICE == "cuda")
    )

    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=(DEVICE == "cuda")
    )

    model = get_model(model_name).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = DiceBCELoss()

    use_amp = DEVICE == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_dice = 0.0
    save_path = f"best_{model_name}_{dataset_name}.pth"

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        model.eval()
        val_dice = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)

                outputs = model(images)

                val_dice += diceScore(outputs, masks).item()
                val_iou += iouScore(outputs, masks).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_dice = val_dice / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        print(
            f"Epoch [{epoch+1}/{epochs}] - "
            f"Loss: {avg_train_loss:.4f} - "
            f"Val Dice: {avg_val_dice:.4f} - "
            f"Val IoU: {avg_val_iou:.4f}"
        )

        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(model.state_dict(), save_path)
    
    print(f"Best model saved to {save_path} with Dice Score: {best_dice:.4f}")
    return save_path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["unet", "resunet"])
    parser.add_argument("--dataset", required=True, choices=["shcxr", "jsrt"])
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=256)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(
        img_dir=args.image_dir,
        mask_dir=args.mask_dir,
        model_name=args.model,
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.image_size
    )