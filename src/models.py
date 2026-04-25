from src.unet import UNet
from src.resunet import ResUNet

def get_model(model_name):
    """
    Model selector

    Valid model names:
    - "unet"
    - "resunet"
    """
    model_name = model_name.lower()

    if model_name == "unet":
        return UNet(in_channels=1, out_channels=1)

    if model_name == "resunet":
        return ResUNet(in_channels=1, out_channels=1)

    raise ValueError(f"Unknown model name: {model_name}")