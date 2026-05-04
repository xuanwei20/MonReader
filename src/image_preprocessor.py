import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import rotate


class ImagePreprocessor:
    def __init__(self, target_size=(108, 198)):
        self.target_size = target_size
        
    def preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert BGR to RGB (OpenCV loads as BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img_resized = cv2.resize(img_rgb, self.target_size)
        
        # Convert to grayscale
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        
        # Normalize pixel values to [0, 1]
        img_normalized = img_gray / 255.0
        
        # Add channel dimension for CNN (grayscale has 1 channel)
        img_final = np.expand_dims(img_normalized, axis=-1)
        
        return img_final
    
    def augment_image(self, image):
        """Apply augmentations to a single image"""
        augmented_images = []
        
        # Original
        augmented_images.append(image)
        
        # Horizontal flip
        horizontal_flip = np.fliplr(image)
        augmented_images.append(horizontal_flip)
        
        # Vertical flip
        vertical_flip = np.flipud(image)
        augmented_images.append(vertical_flip)
        
        # Random brightness adjustment
        brightness_factor = np.random.uniform(0.7, 1.3)
        brightness_adjusted = np.clip(image * brightness_factor, 0, 1)
        augmented_images.append(brightness_adjusted)
        
        # Small rotation (-10 to 10 degrees)
        angle = np.random.uniform(-10, 10)
        rotated = rotate(image.squeeze(), angle, reshape=False, mode='nearest')
        rotated = np.expand_dims(rotated, axis=-1)
        augmented_images.append(rotated)
        
        return augmented_images
    
    def prepare_dataset(self, image_paths, labels, augment=False):
        """Prepare full dataset for training"""
        print(f"Preprocessing Dataset...")
        
        X = []
        y = []
        
        for idx, (img_path, label) in enumerate(zip(image_paths, labels)):
            if idx % 100 == 0:
                print(f"  Processing image {idx}/{len(image_paths)}")
            
            # Preprocess image
            processed_img = self.preprocess_image(img_path)
            
            if processed_img is not None:
                X.append(processed_img)
                y.append(1 if label == 'flipping' else 0)
                
                # Apply augmentations if requested
                if augment:
                    augmented_imgs = self.augment_image(processed_img)
                    for aug_img in augmented_imgs[1:]:  # Skip original as we already added it
                        X.append(aug_img)
                        y.append(1 if label == 'flipping' else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Preprocessing complete!")
        print(f"  Final dataset size: {len(X)} images")
        print(f"  Image shape: {X[0].shape}")
        
        return X, y
    
    def visualize_preprocessing_steps(self, image_path):
        """Visualize preprocessing steps on a sample image"""
        print(f"\nPreprocessing Visualization...")
        
        # Original image
        original = cv2.imread(image_path)
        if original is None:
            print("  Could not load image for visualization")
            return
            
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Resized
        resized = cv2.resize(original_rgb, self.target_size)
        
        # Grayscale
        grayscale = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        
        # Normalized
        normalized = grayscale / 255.0
        
        # Augmented versions
        augmented = self.augment_image(np.expand_dims(normalized, axis=-1))
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        axes[0, 0].imshow(original_rgb)
        axes[0, 0].set_title('Original Image', fontsize=10, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(resized)
        axes[0, 1].set_title(f'Resized ({self.target_size[0]}x{self.target_size[1]})', fontsize=10, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(grayscale, cmap='gray')
        axes[0, 2].set_title('Grayscale', fontsize=10, fontweight='bold')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(augmented[1].squeeze(), cmap='gray')
        axes[1, 0].set_title('Horizontal Flip', fontsize=10, fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(augmented[2].squeeze(), cmap='gray')
        axes[1, 1].set_title('Vertical Flip', fontsize=10, fontweight='bold')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(augmented[4].squeeze(), cmap='gray')
        axes[1, 2].set_title('Rotated', fontsize=10, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.suptitle('Image Preprocessing and Augmentation Pipeline', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('figures/preprocessing_pipeline.png', dpi=100)
        plt.show()