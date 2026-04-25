import os
from pathlib import Path
from PIL import Image

IMG_SIZE = 256
VALID_EXT = [".png", ".jpg", ".jpeg"]

def clean_stem(path):
    stem = path.stem
    return stem.replace("_mask", "").replace("-mask", "").replace(" mask", "")

def get_image_files(folder):
    folder = Path(folder)
    return sorted([
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXT
    ])

def preprocess_dataset(
    dataset_name,
    raw_image_dir,
    raw_mask_dir,
    out_image_dir,
    out_mask_dir,
    img_size=IMG_SIZE,
    mask_threshold=10
):
    raw_image_dir = Path(raw_image_dir)
    raw_mask_dir = Path(raw_mask_dir)
    out_image_dir = Path(out_image_dir)
    out_mask_dir = Path(out_mask_dir)

    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    img_files = get_image_files(raw_image_dir)
    mask_files = get_image_files(raw_mask_dir)

    img_dict = {clean_stem(f): f for f in img_files}
    mask_dict = {clean_stem(f): f for f in mask_files}

    matched_ids = sorted(set(img_dict.keys()) & set(mask_dict.keys()))
    missing_masks = sorted(set(img_dict.keys()) - set(mask_dict.keys()))
    missing_images = sorted(set(mask_dict.keys()) - set(img_dict.keys()))

    print("=" * 60)
    print(f"Preprocessing dataset : {dataset_name}")
    print(f"Total raw images found : {len(img_files)}")
    print(f"Total raw masks found : {len(mask_files)}")
    print(f"Matched image-mask pairs : {len(matched_ids)}")
    print(f"Images without masks ignored: {len(missing_masks)}")
    print(f"Masks without images ignored: {len(missing_images)}")

    if len(matched_ids) == 0:
        raise ValueError(f"No matching image-mask pairs found for {dataset_name}. Check your directories and naming conventions.")

    for img_id in matched_ids:
        img_path = img_dict[img_id]
        mask_path = mask_dict[img_id]

        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        img_resized = img.resize((img_size, img_size))
        mask_resized = mask.resize((img_size, img_size))

        mask_binary = mask_resized.point(lambda p: 255 if p > mask_threshold else 0)

        out_img_path = out_image_dir / f"{img_id}.png"
        out_mask_path = out_mask_dir / f"{img_id}.png"

        img_resized.save(out_img_path)
        mask_binary.save(out_mask_path)
    
    print(f"Saved preprocessed image to {out_image_dir}")
    print(f"Saved preprocessed mask to {out_mask_dir}")
    print(f"{dataset_name} preprocessing complete")

def preprocess_all():
    preprocess_dataset(
        dataset_name="JSRT",
        raw_image_dir="../raw_data/jsrt/images",
        raw_mask_dir="../raw_data/jsrt/masks",
        out_image_dir="../data/jsrt/images",
        out_mask_dir="../data/jsrt/masks",
        img_size=IMG_SIZE,
        mask_threshold=10
    )

    preprocess_dataset(
        dataset_name="SHCXR",
        raw_image_dir="../raw_data/shcxr/images",
        raw_mask_dir="../raw_data/shcxr/masks",
        out_image_dir="../data/shcxr/images",
        out_mask_dir="../data/shcxr/masks",
        img_size=IMG_SIZE,
        mask_threshold=10
    )


if __name__ == "__main__":
    preprocess_all()