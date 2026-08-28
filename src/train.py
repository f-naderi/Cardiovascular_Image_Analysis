import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.dataset import ARCADEDataset
from src.model import UNet


# ==========================================================
# Configuration
# ==========================================================

DATASET_ROOT = "arcade/stenosis"
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable FP16 Mixed Precision only when CUDA is available
USE_AMP = torch.cuda.is_available()

CHECKPOINT_DIR = "results"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")



# Dice Score
def dice_score(pred, target, smooth=1e-6):

    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    pred = pred.view(pred.size(0), -1)

    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)

    dice = ((2.0 * intersection + smooth)/(pred.sum(dim=1)+target.sum(dim=1)+smooth))

    return dice.mean()



# Dice Loss
def dice_loss(pred, target, smooth=1e-6):

    pred = torch.sigmoid(pred)
    pred = pred.view(pred.size(0), -1)

    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)

    dice = ((2.0 * intersection + smooth)/(pred.sum(dim=1)+target.sum(dim=1)+smooth))

    return 1.0 - dice.mean()



# Combined BCE + Dice Loss
class BCEDiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):

        bce = self.bce(pred, target)
        dice = dice_loss(pred, target)

        return bce + dice



# Train One Epoch
def train_one_epoch(model, loader, optimizer, criterion, device, scaler):

    model.train()

    running_loss = 0.0
    running_dice = 0.0

    progress_bar = tqdm(loader, desc="Training", leave=True)

    for images, masks in progress_bar:

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # Clear gradients
        optimizer.zero_grad(set_to_none=True)

        # --------------------------------------------------
        # FP16 Mixed Precision Forward Pass
        # --------------------------------------------------

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):

            outputs = model(images)
            loss = criterion(outputs, masks)

        # --------------------------------------------------
        # Mixed Precision Backward Pass
        # --------------------------------------------------

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # --------------------------------------------------
        # Dice
        # --------------------------------------------------

        dice = dice_score(outputs,masks).item()
        running_loss += loss.item()
        running_dice += dice

        # Progress bar
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice:.4f}")

    epoch_loss = (running_loss / len(loader))
    epoch_dice = (running_dice / len(loader))

    return epoch_loss, epoch_dice


# ==========================================================
# Validation
# ==========================================================

def validate(model, loader, criterion, device):

    model.eval()

    running_loss = 0.0
    running_dice = 0.0

    progress_bar = tqdm(loader, desc="Validation", leave=True)

    with torch.no_grad():

        for images, masks in progress_bar:

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # ----------------------------------------------
            # FP16 Mixed Precision Validation
            # ----------------------------------------------

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):

                outputs = model(images)
                loss = criterion(outputs, masks)

            dice = dice_score(outputs, masks).item()

            running_loss += loss.item()
            running_dice += dice

            progress_bar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice:.4f}")

    epoch_loss = (running_loss / len(loader))
    epoch_dice = (running_dice / len(loader))

    return epoch_loss, epoch_dice


# ==========================================================
# Plot Curves
# ==========================================================

def plot_curves(train_losses, val_losses, train_dices, val_dices):

    # ------------------------------------------------------
    # Loss Curve
    # ------------------------------------------------------

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, "loss_curve.png"), dpi=300)
    plt.close()

    # ------------------------------------------------------
    # Dice Curve
    # ------------------------------------------------------

    plt.figure(figsize=(8, 5))
    plt.plot(train_dices, label="Train Dice")
    plt.plot(val_dices, label="Validation Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Training and Validation Dice")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, "dice_curve.png"), dpi=300)
    plt.close()



def main():

    print(f"Device: {DEVICE}")
    print(f"Mixed Precision FP16: {'Enabled' if USE_AMP else 'Disabled'}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")

    # Results directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ======================================================
    # Dataset
    # ======================================================

    train_dataset = ARCADEDataset(root_dir=DATASET_ROOT, split="train", image_size=IMAGE_SIZE)
    val_dataset = ARCADEDataset(root_dir=DATASET_ROOT, split="val", image_size=IMAGE_SIZE)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # ======================================================
    # DataLoader
    # ======================================================

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

    # Model
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)

    # Loss
    criterion = BCEDiceLoss()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # FP16 GradScaler
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    # ======================================================
    # Training History
    # ======================================================

    train_losses = []
    val_losses = []

    train_dices = []
    val_dices = []

    best_val_dice = 0.0

    # ======================================================
    # Training Loop
    # ======================================================

    for epoch in range(NUM_EPOCHS):

        print()
        print("-" * 50)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
        print("-" * 50)

        # --------------------------------------------------
        # Training
        # --------------------------------------------------

        train_loss, train_dice = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE,
            scaler=scaler
        )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        val_loss, val_dice = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=DEVICE
        )

        # --------------------------------------------------
        # Save History
        # --------------------------------------------------

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(train_dice)
        val_dices.append(val_dice)

        # --------------------------------------------------
        # Epoch Results
        # --------------------------------------------------

        print()
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Dice: {train_dice:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val Dice:   {val_dice:.4f}")

        # --------------------------------------------------
        # Save Best Model
        # --------------------------------------------------

        if val_dice > best_val_dice:

            best_val_dice = val_dice

            torch.save(model.state_dict(), BEST_MODEL_PATH)

            print()
            print(f"Best model saved")
            print(f"Best Val Dice: {best_val_dice:.4f}")


    # Plot Curves
    plot_curves(train_losses=train_losses, val_losses=val_losses, train_dices=train_dices, val_dices=val_dices)

    # ======================================================
    # Final Results
    # ======================================================

    print()
    print("-" * 50)
    print("Training completed.")
    print("-" * 50)
    print(f"Best Validation Dice: {best_val_dice:.4f}")
    print(f"Best model: {BEST_MODEL_PATH}")
    print(f"Loss curve: {CHECKPOINT_DIR}/loss_curve.png")
    print(f"Dice curve: {CHECKPOINT_DIR}/dice_curve.png")



# Run
if __name__ == "__main__":
    main()