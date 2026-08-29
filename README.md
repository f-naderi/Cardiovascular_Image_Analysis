```markdown
# Coronary Stenosis Segmentation with U-Net

A medical image segmentation project using a U-Net convolutional neural network to segment coronary artery stenosis in X-ray angiography images from the ARCADE dataset.

## Features

- Coronary stenosis segmentation from X-ray angiography
- COCO-style annotation parsing and binary mask generation
- PyTorch + FP16/CUDA training
- Dice-based evaluation and inference


## Project Structure

Cardiovascular_Image_Analysis/
├── arcade/                     # ARCADE dataset
├── src/                        
│   ├── dataset.py              # Data loading pipeline
│   ├── model.py                # U-Net architecture
│   ├── train.py                # Training pipeline
│   └── inference.py            # Model inference
│
├── results/
│   ├── best_model.pth          # Best validation checkpoint
│   ├── loss_curve.png          # Training/validation loss
│   ├── dice_curve.png          # Training/validation Dice
│   └── predictions/            # Model predictions
│
├── Segmentation.ipynb          # Interactive Jupyter notebook
└── requirements.txt            # Python dependencies


## Installation

```bash
git clone https://github.com/f-naderi/Cardiovascular_Image_Analysis.git
cd Cardiovascular_Image_Analysis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Model Architecture:

* Architecture: U-Net
* Loss: Combined segmentation loss
* Evaluation metric: Dice coefficient
* Optimizer: AdamW
* Mixed Precision: FP16

## Training Details

* Dataset: ARCADE
* Train/Val/Test: 1000 / 200 / 300
* Image Size: 512×512
* Batch Size: 8
* Epochs: 50

   
## Requirements

* Python 3.10+
* PyTorch
* Torchvision
* NumPy
* Pillow
* Matplotlib
* tqdm

## References

* ARCADE: Annotated Coronary Artery Disease dataset
* Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.

```