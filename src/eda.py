import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2


class EDA:
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.train_image_paths = []
        self.train_labels = []
        self.test_image_paths = []
        self.test_labels = []
        
    def load_and_analyze(self):
        print("="*60)
        print("PART 1: EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        train_flip_path = self.data_path / "training" / "flip"
        train_notflip_path = self.data_path / "training" / "notflip"
        
        test_flip_path = self.data_path / "testing" / "flip"
        test_notflip_path = self.data_path / "testing" / "notflip"
        
        print("Loading Training Data...")
        for img_path in train_flip_path.glob('*.jpg'):
            self.train_image_paths.append(str(img_path))
            self.train_labels.append('flipping')
        for img_path in train_flip_path.glob('*.png'):
            self.train_image_paths.append(str(img_path))
            self.train_labels.append('flipping')
            
        for img_path in train_notflip_path.glob('*.jpg'):
            self.train_image_paths.append(str(img_path))
            self.train_labels.append('not_flipping')
        for img_path in train_notflip_path.glob('*.png'):
            self.train_image_paths.append(str(img_path))
            self.train_labels.append('not_flipping')
        
        print("Loading Testing Data...")
        for img_path in test_flip_path.glob('*.jpg'):
            self.test_image_paths.append(str(img_path))
            self.test_labels.append('flipping')
        for img_path in test_flip_path.glob('*.png'):
            self.test_image_paths.append(str(img_path))
            self.test_labels.append('flipping')
            
        for img_path in test_notflip_path.glob('*.jpg'):
            self.test_image_paths.append(str(img_path))
            self.test_labels.append('not_flipping')
        for img_path in test_notflip_path.glob('*.png'):
            self.test_image_paths.append(str(img_path))
            self.test_labels.append('not_flipping')
        
        print(f"\nDataset Statistics:")
        print(f"Training set total: {len(self.train_image_paths)} images")
        print(f"  - Flipping: {self.train_labels.count('flipping')} images")
        print(f"  - Not Flipping: {self.train_labels.count('not_flipping')} images")
        print(f"Testing set total: {len(self.test_image_paths)} images")
        print(f"  - Flipping: {self.test_labels.count('flipping')} images")
        print(f"  - Not Flipping: {self.test_labels.count('not_flipping')} images")
        
        # Visualize class distribution
        self.visualize_class_distribution()
        
        # Analyze sample images
        self.analyze_sample_images()
        
        # Image properties analysis
        self.analyze_image_properties()
        
        return (self.train_image_paths, self.train_labels, 
                self.test_image_paths, self.test_labels)
    
    def visualize_class_distribution(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Training set distribution
        train_unique, train_counts = np.unique(self.train_labels, return_counts=True)
        ax1.bar(train_unique, train_counts, color=['#FF6B6B', '#4ECDC4'])
        ax1.set_title('Training Set Class Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Class', fontsize=10)
        ax1.set_ylabel('Number of Images', fontsize=10)
        for i, v in enumerate(train_counts):
            ax1.text(i, v + 10, str(v), ha='center', fontsize=10, fontweight='bold')
        
        # Testing set distribution
        test_unique, test_counts = np.unique(self.test_labels, return_counts=True)
        ax2.bar(test_unique, test_counts, color=['#FF6B6B', '#4ECDC4'])
        ax2.set_title('Testing Set Class Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Class', fontsize=10)
        ax2.set_ylabel('Number of Images', fontsize=10)
        for i, v in enumerate(test_counts):
            ax2.text(i, v + 5, str(v), ha='center', fontsize=10, fontweight='bold')
        
        plt.suptitle('Dataset Class Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('figures/class_distribution.png', dpi=100)
        plt.show()
    
    def analyze_sample_images(self):
        print(f"\nSample Image Analysis...")
        
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        axes = axes.ravel()
        
        # Get sample images from training set
        flip_samples = [img for img, lbl in zip(self.train_image_paths, self.train_labels) 
                       if lbl == 'flipping'][:4]
        notflip_samples = [img for img, lbl in zip(self.train_image_paths, self.train_labels) 
                          if lbl == 'not_flipping'][:4]
        
        for i, img_path in enumerate(flip_samples):
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i].imshow(img_rgb)
                axes[i].set_title(f"Flipping - Sample {i+1}", fontsize=10)
                axes[i].axis('off')
        
        for i, img_path in enumerate(notflip_samples):
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[4+i].imshow(img_rgb)
                axes[4+i].set_title(f"Not Flipping - Sample {i+1}", fontsize=10)
                axes[4+i].axis('off')
        
        plt.suptitle('Sample Images from Training Set', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('figures/sample_images.png', dpi=100)
        plt.show()
    
    def analyze_image_properties(self):
        """Analyze image dimensions, color channels, etc."""
        print(f"\nImage Properties Analysis...")
        
        dimensions = []
        color_modes = []
        
        # Analyze first 50 training images
        for img_path in self.train_image_paths[:50]:
            img = cv2.imread(img_path)
            if img is not None:
                dimensions.append(img.shape[:2])
                if len(img.shape) == 3:
                    color_modes.append('RGB' if img.shape[2] == 3 else 'RGBA')
                else:
                    color_modes.append('Grayscale')
        
        if not dimensions:
            print("  No valid images found for analysis")
            return
        
        # Dimension analysis
        heights, widths = zip(*dimensions)
        print(f"  Image Height Range: {min(heights)} - {max(heights)} pixels")
        print(f"  Image Width Range: {min(widths)} - {max(widths)} pixels")
        print(f"  Average Height: {np.mean(heights):.0f} pixels")
        print(f"  Average Width: {np.mean(widths):.0f} pixels")
        
        # Color mode distribution
        unique_modes, mode_counts = np.unique(color_modes, return_counts=True)
        print(f"\n  Color Mode Distribution:")
        for mode, count in zip(unique_modes, mode_counts):
            print(f"    {mode}: {count} images")
        
        # Plot dimension distribution
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(heights, bins=20, alpha=0.7, color='#FF6B6B', edgecolor='black')
        plt.xlabel('Height (pixels)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Image Heights', fontsize=12, fontweight='bold')
        
        plt.subplot(1, 2, 2)
        plt.hist(widths, bins=20, alpha=0.7, color='#4ECDC4', edgecolor='black')
        plt.xlabel('Width (pixels)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Image Widths', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('figures/image_dimensions.png', dpi=100)
        plt.show()