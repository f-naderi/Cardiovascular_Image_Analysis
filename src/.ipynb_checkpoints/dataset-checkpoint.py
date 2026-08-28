import os
import json
import random

from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class ARCADEDataset(Dataset):

    def __init__(self, root_dir, split="train", image_size=(512, 512), augment=False):

        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size

        # Augmentation is enabled only for training data
        self.augment = augment and split == "train"

        # ---------------------------------------------------
        # Paths
        # ---------------------------------------------------

        self.image_dir = os.path.join(root_dir, split, "images")
        self.annotation_dir = os.path.join(root_dir, split, "annotations")

        # ---------------------------------------------------
        # Find JSON file
        # ---------------------------------------------------

        json_files = [
            f for f in os.listdir(self.annotation_dir)
            if f.lower().endswith(".json")
        ]


        self.annotation_path = os.path.join(
            self.annotation_dir,
            json_files[0]
        )

        # ---------------------------------------------------
        # Load COCO annotations
        # ---------------------------------------------------

        with open(self.annotation_path, "r") as f:
            data = json.load(f)

        self.images_info = data["images"]
        self.annotations = data["annotations"]
        self.categories = data["categories"]

        # ---------------------------------------------------
        # Map image_id -> image information
        # ---------------------------------------------------

        self.image_info = {
            image["id"]: image
            for image in self.images_info
        }

        # ---------------------------------------------------
        # Map image_id -> annotations
        # ---------------------------------------------------

        self.annotations_by_image = {}

        for annotation in self.annotations:

            image_id = annotation["image_id"]

            if image_id not in self.annotations_by_image:
                self.annotations_by_image[image_id] = []

            self.annotations_by_image[image_id].append(annotation)

        # ---------------------------------------------------
        # Keep only images that actually exist
        # ---------------------------------------------------

        available_images = set(os.listdir(self.image_dir))

        self.images_info = [
            image
            for image in self.images_info
            if image["file_name"] in available_images
        ]


    def __len__(self):
        return len(self.images_info)


    # =======================================================
    # DATA AUGMENTATION
    # =======================================================

    def apply_augmentation(self, image, mask):

        # ---------------------------------------------------
        # 1. Horizontal Flip
        # ---------------------------------------------------

        if random.random() < 0.5:

            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # ---------------------------------------------------
        # 2. Small Rotation
        # ---------------------------------------------------

        if random.random() < 0.5:

            angle = random.uniform(-10.0, 10.0)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)

        # ---------------------------------------------------
        # 3. Small Translation + Scaling
        # ---------------------------------------------------

        if random.random() < 0.3:

            max_translate = 10

            translate_x = random.randint(-max_translate, max_translate)
            translate_y = random.randint(-max_translate, max_translate)

            scale = random.uniform(0.95, 1.05)

            # Same transformation parameters
            # are used for image and mask.

            image = TF.affine(
                image,
                angle=0,
                translate=[
                    translate_x,
                    translate_y
                ],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0
            )

            mask = TF.affine(
                mask,
                angle=0,
                translate=[
                    translate_x,
                    translate_y
                ],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0
            )

        # ---------------------------------------------------
        # 4. Brightness
        # ---------------------------------------------------

        if random.random() < 0.3:

            brightness_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_brightness(image, brightness_factor)

        # ---------------------------------------------------
        # 5. Contrast
        # ---------------------------------------------------

        if random.random() < 0.3:

            contrast_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_contrast(image, contrast_factor)

        # ---------------------------------------------------
        # 6. Gaussian Noise
        # ---------------------------------------------------

        if random.random() < 0.2:

            image = TF.to_tensor(image)

            noise = (random.uniform(0.01, 0.02) * image.new_empty(image.shape).normal_())
            image = image + noise
            image = image.clamp(0.0, 1.0)
            image = TF.to_pil_image(image)

        return image, mask


    # =======================================================
    # GET ITEM
    # =======================================================

    def __getitem__(self, idx):

        image_info = self.images_info[idx]
        image_id = image_info["id"]
        file_name = image_info["file_name"]
        image_path = os.path.join(self.image_dir, file_name)

        # ---------------------------------------------------
        # Load image
        # ---------------------------------------------------

        image = Image.open(image_path).convert("L")

        # ---------------------------------------------------
        # Create binary segmentation mask
        # ---------------------------------------------------

        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        image_annotations = self.annotations_by_image.get(image_id, [])

        for annotation in image_annotations:

            segmentation = annotation.get("segmentation", [])

            # Each segmentation can contain one or more polygons
            for polygon in segmentation:

                if len(polygon) < 6:
                    continue

                points = [
                    (
                        polygon[i],
                        polygon[i + 1]
                    )
                    for i in range(0, len(polygon), 2)
                ]

                draw.polygon(points, fill=255)

        # ---------------------------------------------------
        # Resize
        # ---------------------------------------------------

        image = TF.resize(image, self.image_size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.image_size, interpolation=InterpolationMode.NEAREST)


        # Apply augmentation ONLY for training
        if self.augment:
            image, mask = self.apply_augmentation(image, mask)

        # ---------------------------------------------------
        # Convert to Tensor
        # ---------------------------------------------------

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # Normalize image to [-1, 1]
        image = image * 2.0 - 1.0

        # Binary mask
        mask = (mask > 0.5).float()

        return image, mask