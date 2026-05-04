from eda import EDA
from image_preprocessor import ImagePreprocessor
from text_extractor import TextExtractorAndReader
from model_trainer import PageFlipCNN_PyTorch
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator

import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


def main():
    print("="*80)
    print("MONREADER PAGE FLIP DETECTION SYSTEM")
    print("CNN-Based Classification Pipeline (PyTorch)")
    print("="*80)
    
    DATA_PATH = Path(r"C:\Users\User\Desktop\python vscode\project_4\data\raw")
    TARGET_SIZE = (108, 198)
    BATCH_SIZE = 32
    EPOCHS = 50
    VALIDATION_SPLIT = 0.2
    
    # Step 1: Exploratory Data Analysis
    eda = EDA(DATA_PATH)
    train_paths, train_labels, test_paths, test_labels = eda.load_and_analyze()
    
    if len(train_paths) == 0:
        print("\nError: No training images found! Please check your data path.")
        return
    
    # Step 2: OCR and Text-to-Speech Feature with Translation
    print("\n" + "="*60)
    print("STEP 2: TEXT EXTRACTION, TRANSLATION AND AUDIO FEATURE")
    print("="*60)
    
    # Ask user if they want to try OCR on an image
    print("\nDo you want to extract text, translate it to English, and read it aloud?")
    print("Options:")
    print("  1. Use one of the test images")
    print("  2. Provide your own image path")
    print("  3. Skip this feature")
    
    choice = input("Enter your choice (1/2/3): ").strip()
    
    if choice in ['1', '2']:
        if choice == '1':
            if len(test_paths) > 0:
                selected_image = test_paths[0]
                print(f"\nSelected test image: {selected_image}")
            else:
                print("No test images available")
                selected_image = None
        else:  # choice == '2'
            selected_image = input("Enter the full path to your image: ").strip()
            if not Path(selected_image).exists():
                print(f"Image not found: {selected_image}")
                selected_image = None
        
        if selected_image:
            # Ask for Tesseract path if on Windows
            tesseract_path = None
            if os.name == 'nt':
                print("\nFor Windows users: If Tesseract is not in default path,")
                print("   please provide the full path to tesseract.exe")
                print("   Example: C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
                custom_path = input("Enter Tesseract path (or press Enter to use default): ").strip()
                if custom_path:
                    tesseract_path = custom_path
            
            # Ask for source language
            print("\nSource language options:")
            print("  tr - Turkish")
            print("  en - English")
            print("  fr - French")
            print("  de - German")
            print("  es - Spanish")
            source_lang = input("Enter source language code (default: tr): ").strip() or 'tr'
            
            # Target language for translation
            target_lang = 'en'  # Fixed to English for speech output
            print(f"Target language for speech: {target_lang} (English)")
            
            # Initialize text extractor with translation
            text_reader = TextExtractorAndReader(
                tesseract_path=tesseract_path,
                source_lang=source_lang,
                target_lang=target_lang
            )
            
            # Ask user if they want preprocessing
            preprocess_choice = input("Apply image preprocessing for better OCR? (y/n): ").strip().lower()
            preprocess = preprocess_choice == 'y'
            
            # Ask if user wants translation
            translate_choice = input(f"Translate from {source_lang} to English? (y/n): ").strip().lower()
            translate = translate_choice == 'y'
            
            # Ask for speech speed
            speed_choice = input("Speak slowly? (y/n): ").strip().lower()
            slow = speed_choice == 'y'
            
            # Ask if user wants to save audio file
            save_audio_choice = input("Save audio to file? (y/n): ").strip().lower()
            save_audio = save_audio_choice == 'y'
            
            audio_dir = None
            if save_audio:
                audio_dir = input("Enter directory to save audio files (press Enter for 'audio_outputs'): ").strip()
                if not audio_dir:
                    audio_dir = "audio_outputs"
            
            # Process the image with translation and audio saving
            result = text_reader.process_image_with_audio(
                selected_image, 
                preprocess=preprocess,
                translate=translate,
                target_lang=target_lang,
                slow=slow,
                save_translated=True,
                save_audio=save_audio,
                audio_output_dir=audio_dir
            )
            
            # Show the result summary
            if result:
                print("\n" + "="*60)
                print("OCR PROCESSING COMPLETE - SUMMARY")
                print("="*60)
                print(f"Text file: {result['text_file']}")
                if result['audio_file']:
                    print(f"Audio file: {result['audio_file']}")
                if result['translated_text']:
                    print(f"Translation: {source_lang} → {target_lang}")
                print(f"Original text length: {len(result['original_text'])} characters")
                if result['translated_text']:
                    print(f"Translated text length: {len(result['translated_text'])} characters")
                print("="*60)
            
            # Show OCR visualization
            if result and result['original_text']:
                show_viz = input("\nShow OCR visualization with bounding boxes? (y/n): ").strip().lower()
                if show_viz == 'y':
                    text_reader.visualize_ocr_result(selected_image)
        else:
            print("Skipping OCR feature...")
    else:
        print("Skipping OCR feature...")
    
    # Step 3: Image Preprocessing for CNN Model
    preprocessor = ImagePreprocessor(target_size=TARGET_SIZE)
    
    if len(train_paths) > 0:
        preprocessor.visualize_preprocessing_steps(train_paths[0])
    
    # Prepare training dataset (with augmentation)
    print("\n" + "="*60)
    print("\nPreparing Training Data...")
    print("="*60)
    X_train_full, y_train_full = preprocessor.prepare_dataset(train_paths, train_labels, augment=True)
    
    # Prepare testing dataset (without augmentation)
    print("\n" + "="*60)
    print("\nPreparing Testing Data...")
    print("="*60)
    X_test, y_test = preprocessor.prepare_dataset(test_paths, test_labels, augment=False)
    
    # Step 4: Split training data into train and validation sets
    print(f"\nSplitting Training Data...")
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=VALIDATION_SPLIT, 
        random_state=42, stratify=y_train_full
    )
    
    print(f"  Training set: {len(X_train)} images")
    print(f"  Validation set: {len(X_val)} images")
    print(f"  Test set: {len(X_test)} images")
    
    # Step 5: Create data loaders
    X_train_tensor = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val).permute(0, 3, 1, 2)
    y_val_tensor = torch.FloatTensor(y_val)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Step 6: Build and train model
    model = PageFlipCNN_PyTorch()
    trainer = ModelTrainer(model, learning_rate=0.001)
    
    # Print model architecture
    print("\nModel Architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    history = trainer.train(train_loader, val_loader, epochs=EPOCHS)
    
    # Plot training history
    trainer.plot_training_history(history)
    
    # Step 7: Evaluation on test set
    print("\n" + "="*60)
    evaluator = ModelEvaluator(model, preprocessor)
    f1_score_final, predictions = evaluator.evaluate(X_test, y_test)
    
    evaluator.visualize_predictions(X_test, y_test, num_samples=8)
    
    # Step 8: Save final model
    print("\n" + "="*60)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': str(model),
        'final_f1_score': f1_score_final,
        'input_size': TARGET_SIZE
    }, 'results/monreader_page_flip_detector_final.pth')
    print("Model saved as 'results/monreader_page_flip_detector_final.pth'")
    
    # Step 9: Test on sample test images
    print("\n" + "="*60)
    print("Testing on sample test images:")
    test_samples = min(3, len(test_paths))
    for i in range(test_samples):
        print(f"\nTest Sample {i+1}:")
        evaluator.predict_single_image(test_paths[i])
    
    
    print("\n" + "="*60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print(f"Final Test F1 Score: {f1_score_final:.4f}")
    print("="*60)
    
    return model, f1_score_final, history

if __name__ == "__main__":
    main()
    # model, f1_score, history = main()
