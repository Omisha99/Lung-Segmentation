import pandas as pd
from evaluate import evaluate_model

# experiments = [
#     ("U-Net", "SHCXR", "SHCXR", "unet", "best_unet_shcxr.pth", "data/shcxr/images", "data/shcxr/masks"),
#     ("U-Net", "SHCXR", "JSRT", "unet", "best_unet_shcxr.pth", "data/jsrt/images", "data/jsrt/masks"),
#     ("U-Net", "JSRT", "JSRT", "unet", "best_unet_jsrt.pth", "data/jsrt/images", "data/jsrt/masks"),
#     ("U-Net", "JSRT", "SHCXR", "unet", "best_unet_jsrt.pth", "data/shcxr/images", "data/shcxr/masks"),

#     ("ResUNet", "SHCXR", "SHCXR", "resunet", "best_resunet_shcxr.pth", "data/shcxr/images", "data/shcxr/masks"),
#     ("ResUNet", "SHCXR", "JSRT", "resunet", "best_resunet_shcxr.pth", "data/jsrt/images", "data/jsrt/masks"),
#     ("ResUNet", "JSRT", "JSRT", "resunet", "best_resunet_jsrt.pth", "data/jsrt/images", "data/jsrt/masks"),
#     ("ResUNet", "JSRT", "SHCXR", "resunet", "best_resunet_jsrt.pth", "data/shcxr/images", "data/shcxr/masks"),
# ]

experiments = [
    ("U-Net", "SHCXR", "SHCXR", "unet", "best_unet_shcxr.pth", "data/shcxr/images", "data/shcxr/masks", "split_unet_shcxr.json"),
    ("U-Net", "SHCXR", "JSRT", "unet", "best_unet_shcxr.pth", "data/jsrt/images", "data/jsrt/masks", None),

    ("U-Net", "JSRT", "JSRT", "unet", "best_unet_jsrt.pth", "data/jsrt/images", "data/jsrt/masks", "split_unet_jsrt.json"),
    ("U-Net", "JSRT", "SHCXR", "unet", "best_unet_jsrt.pth", "data/shcxr/images", "data/shcxr/masks", None),

    ("ResUNet", "SHCXR", "SHCXR", "resunet", "best_resunet_shcxr.pth", "data/shcxr/images", "data/shcxr/masks", "split_resunet_shcxr.json"),
    ("ResUNet", "SHCXR", "JSRT", "resunet", "best_resunet_shcxr.pth", "data/jsrt/images", "data/jsrt/masks", None),

    ("ResUNet", "JSRT", "JSRT", "resunet", "best_resunet_jsrt.pth", "data/jsrt/images", "data/jsrt/masks", "split_resunet_jsrt.json"),
    ("ResUNet", "JSRT", "SHCXR", "resunet", "best_resunet_jsrt.pth", "data/shcxr/images", "data/shcxr/masks", None),
]

def main():
    results = []
    for model_label, trained_on, tested_on, model_name, model_path, image_dir, mask_dir, split_path in experiments:
        print("=" * 60)
        print(f"{model_label}: train {trained_on} -> test {tested_on}")

        dice, iou = evaluate_model(
            img_dir=image_dir,
            mask_dir=mask_dir,
            model_path=model_path,
            model_name=model_name,
            batch_size=8,
            img_size=256,
            split_path=split_path,
            split_name="val"
        )

        results.append({
            "Model": model_label,
            "Trained On": trained_on,
            "Tested On": tested_on,
            "Dice Score": f"{dice:.4f}",
            "IoU Score": f"{iou:.4f}"
        })

    df = pd.DataFrame(results)
    df.to_csv("evaluation_results.csv", index=False)

    print("=" * 60)
    print("Final results:")
    print(df)
    print("Saved results to evaluation_results.csv")


if __name__ == "__main__":
    main()
