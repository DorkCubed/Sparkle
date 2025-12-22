import torch
import os
from sparkle.configs.config import Config
from sparkle.utils import get_project_root

config = Config()
project_root = get_project_root()


def load_losses(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"File not found: {checkpoint_path}")
        return None

    # 1. Load the checkpoint dictionary
    # map_location='cpu' ensures it loads even if you don't have a GPU currently available
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # 2. Access the loss keys defined in your save_checkpoint function
    mlm_loss = checkpoint["accumulated_mlm_loss"]
    sfbo_loss = checkpoint["accumulated_sfbo_loss"]
    epoch = checkpoint["epoch"]

    # 3. Check if they are tensors or floats
    # If they were saved as tensors, use .item() to get the standard Python float
    if torch.is_tensor(mlm_loss):
        mlm_loss = mlm_loss.item()

    if torch.is_tensor(sfbo_loss):
        sfbo_loss = sfbo_loss.item()

    print(f"Loaded Checkpoint from Epoch {epoch}")
    print(f"MLM Loss: {mlm_loss}")
    print(f"SFBO Loss: {sfbo_loss}")

    return mlm_loss, sfbo_loss


path = os.path.join(project_root, "src", "checkpoints", "checkpoint_1_5k.pth")
load_losses(path)