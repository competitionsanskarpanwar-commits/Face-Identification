import os
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
import random

# ================================
# CONFIGURATION
# ================================
INPUT_DIR = "data/processed"  # Raw face images directory
OUTPUT_DIR = "data/augmented"  # Output directory for augmented images

# Target resolution for all images
TARGET_SIZE = 320  # Match the img_size in config.yaml

# Target number of images per class
TARGET_IMAGES_PER_CLASS = 75  # Between 50-100

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# AUGMENTATION PIPELINES FOR FACE IMAGES
# ================================

# Light augmentations (preserve facial features)
light_transform = A.Compose([
    # Small exposure adjustments (±10%)
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.8),
    
    # Small color adjustments
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.6),
    
    # Small rotation (±5 degrees)
    A.Affine(rotate=(-5, 5), p=0.5),
    
    # Small scale changes
    A.Affine(scale=(0.95, 1.05), p=0.3),
    
    # Light blur (simulate different camera focus)
    A.GaussianBlur(blur_limit=(1, 3), p=0.2),
])

# Medium augmentations
medium_transform = A.Compose([
    # Moderate exposure adjustments (±15%)
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.8),
    
    # Color adjustments
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.7),
    
    # Moderate rotation (±10 degrees)
    A.Affine(rotate=(-10, 10), p=0.6),
    
    # Scale and translation
    A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), p=0.4),
    
    # Perspective warp (subtle)
    A.Perspective(scale=(0.01, 0.03), p=0.3),
    
    # Blur variations
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    
    # Random crop
    A.RandomCrop(height=int(TARGET_SIZE * 0.9), width=int(TARGET_SIZE * 0.9), p=0.2),
])

# Strong augmentations (use sparingly)
strong_transform = A.Compose([
    # Larger exposure adjustments (±20%)
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
    
    # Stronger color adjustments
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=25, val_shift_limit=20, p=0.7),
    
    # Larger rotation (±15 degrees)
    A.Affine(rotate=(-15, 15), p=0.6),
    
    # Scale and translation
    A.Affine(scale=(0.85, 1.15), translate_percent=(-0.1, 0.1), p=0.5),
    
    # Perspective warp
    A.Perspective(scale=(0.02, 0.05), p=0.4),
    
    # Various blur effects
    A.GaussianBlur(blur_limit=(3, 5), p=0.4),
    A.MotionBlur(blur_limit=3, p=0.2),
    
    # Random crop
    A.RandomCrop(height=int(TARGET_SIZE * 0.85), width=int(TARGET_SIZE * 0.85), p=0.3),
    
    # Randomly convert to grayscale (10% chance)
    A.ToGray(p=0.1),
])

# ================================
# FACE IMAGE AUGMENTATION FUNCTIONS
# ================================

def augment_face_images(class_dir, class_name, target_count):
    """Augment images in a class directory to reach target count"""
    
    # Get all image files in the class directory
    image_files = [f for f in os.listdir(class_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No images found in {class_dir}")
        return
    
    # Create output directory for this class
    output_class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(output_class_dir, exist_ok=True)
    
    # Calculate how many augmentations we need
    current_count = len(image_files)
    needed_count = target_count - current_count
    
    if needed_count <= 0:
        print(f"Class {class_name} already has {current_count} images (>= {target_count})")
        # Just copy original images to output
        for img_file in image_files:
            src_path = os.path.join(class_dir, img_file)
            dst_path = os.path.join(output_class_dir, img_file)
            cv2.imwrite(dst_path, cv2.imread(src_path))
        return
    
    print(f"Class {class_name}: {current_count} original images, need {needed_count} augmentations")
    
    # Copy original images first
    for img_file in image_files:
        src_path = os.path.join(class_dir, img_file)
        dst_path = os.path.join(output_class_dir, img_file)
        cv2.imwrite(dst_path, cv2.imread(src_path))
    
    # Generate augmentations
    augmentation_count = 0
    transform_weights = [0.5, 0.3, 0.2]  # Light, medium, strong weights
    
    while augmentation_count < needed_count:
        # Randomly select an original image
        original_img_file = random.choice(image_files)
        img_path = os.path.join(class_dir, original_img_file)
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # Resize to target size first
        image = cv2.resize(image, (TARGET_SIZE, TARGET_SIZE))
        
        # Randomly select augmentation strength
        transform_choice = random.choices(
            [light_transform, medium_transform, strong_transform], 
            weights=transform_weights
        )[0]
        
        # Apply augmentation
        augmented = transform_choice(image=image)
        aug_img = augmented['image']
        
        # Generate output filename
        base_name = os.path.splitext(original_img_file)[0]
        aug_filename = f"{base_name}_aug_{augmentation_count:03d}.jpg"
        aug_path = os.path.join(output_class_dir, aug_filename)
        
        # Save augmented image
        cv2.imwrite(aug_path, aug_img)
        augmentation_count += 1
    
    print(f"Generated {augmentation_count} augmentations for {class_name}")
    return current_count + augmentation_count

# ================================
# MAIN AUGMENTATION PROCESS
# ================================

def main():
    print("Starting face image augmentation...")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Target images per class: {TARGET_IMAGES_PER_CLASS}")
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(INPUT_DIR) 
                  if os.path.isdir(os.path.join(INPUT_DIR, d))]
    
    if not class_dirs:
        print(f"No class directories found in {INPUT_DIR}")
        return
    
    print(f"Found {len(class_dirs)} classes: {class_dirs}")
    
    total_images = 0
    results = {}
    
    # Process each class
    for class_name in tqdm(class_dirs, desc="Processing classes"):
        class_dir = os.path.join(INPUT_DIR, class_name)
        
        try:
            count = augment_face_images(class_dir, class_name, TARGET_IMAGES_PER_CLASS)
            results[class_name] = count
            total_images += count
        except Exception as e:
            print(f"Error processing {class_name}: {e}")
            results[class_name] = 0
    
    # Print summary
    print("\n" + "="*50)
    print("AUGMENTATION COMPLETE!")
    print("="*50)
    print(f"Total images generated: {total_images}")
    print("\nPer-class results:")
    for class_name, count in results.items():
        print(f"  {class_name}: {count} images")
    
    print(f"\nOutput saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()