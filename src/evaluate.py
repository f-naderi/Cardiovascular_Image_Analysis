import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.dataset import ARCADEDataset
from src.model import UNet


# =========================================================
# Configuration
# =========================================================

DATASET_ROOT = "arcade/stenosis"
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 8
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "results/best_model.pth"
OUTPUT_DIR = "results/predictions"
NUM_IMAGES_TO_SAVE = 10
THRESHOLD = 0.5


# =========================================================
# Load Model
# =========================================================

def load_model():

    model = UNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    return model


# =========================================================
# Dice Score
# =========================================================

def dice_score(prediction, target, smooth=1e-6):

    prediction = prediction.float()
    target = target.float()

    prediction = prediction.view(prediction.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (prediction * target).sum(dim=1)

    dice = ((2.0 * intersection + smooth)/(prediction.sum(dim=1)+target.sum(dim=1)+smooth))

    return dice.mean().item()


# =========================================================
# Save Prediction Visualization
# =========================================================

def save_visualization(image, mask, prediction, index):

    image = image.squeeze().cpu()
    mask = mask.squeeze().cpu()
    prediction = prediction.squeeze().cpu()

    # Convert image from [-1, 1] to [0, 1]
    image = (image + 1.0) / 2.0
    plt.figure(figsize=(15, 5))

    # -----------------------------------------------------
    # Input
    # -----------------------------------------------------

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Input Image")
    plt.axis("off")

    # -----------------------------------------------------
    # Ground Truth
    # -----------------------------------------------------

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    plt.subplot(1, 3, 3)
    plt.imshow(prediction, cmap="gray")
    plt.title("Prediction")
    plt.axis("off")
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f"prediction_{index:03d}.png")

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()



def main():

    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load Test Dataset
    test_dataset = ARCADEDataset(root_dir=DATASET_ROOT, split="test", image_size=IMAGE_SIZE, augment=False)
    print(f"Test samples: {len(test_dataset)}")

    # DataLoader
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

    # Load Model
    model = load_model()

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    running_dice = 0.0
    num_batches = 0
    saved_images = 0

    with torch.no_grad():

        progress_bar = tqdm(test_loader, desc="Evaluation", leave=True)

        for images, masks in progress_bar:

            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            # ---------------------------------------------
            # FP16 Evaluation
            # ---------------------------------------------

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):

                outputs = model(images)

            # Convert logits to probabilities
            probabilities = torch.sigmoid(outputs)

            # Binary segmentation
            predictions = (probabilities >= THRESHOLD).float()

            # ---------------------------------------------
            # Dice
            # ---------------------------------------------

            batch_dice = dice_score(predictions, masks)
            running_dice += batch_dice
            num_batches += 1
            progress_bar.set_postfix(dice=f"{batch_dice:.4f}")

            # ---------------------------------------------
            # Save visualizations
            # ---------------------------------------------

            for i in range(images.size(0)):

                if saved_images >= NUM_IMAGES_TO_SAVE:
                    break

                save_visualization(images[i], masks[i], predictions[i], saved_images)
                saved_images += 1

    # -----------------------------------------------------
    # Final Dice
    # -----------------------------------------------------

    mean_dice = (running_dice / num_batches)

    print()
    print("-" * 50)
    print("Evaluation completed.")
    print("-" * 50)
    print(f"Test Dice: {mean_dice:.4f}")
    print(f"Saved predictions: {saved_images}")
    print(f"Output directory: {OUTPUT_DIR}")


# Run
if __name__ == "__main__":
    main()