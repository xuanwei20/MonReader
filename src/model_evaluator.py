import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import torch

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")    

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class ModelEvaluator:
    
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.model.eval()
    
    def evaluate(self, X_test, y_test):

        print(f"\nModel Evaluation on Test Set:")
        
        X_test_tensor = torch.FloatTensor(X_test).permute(0, 3, 1, 2).to(device)
        
        with torch.no_grad():
            y_pred_prob = self.model(X_test_tensor).cpu().numpy()
        
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\nTest Set Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Not Flipping', 'Flipping']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Not Flipping', 'Flipping'],
                    yticklabels=['Not Flipping', 'Flipping'])
        plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.tight_layout()
        plt.savefig('figures/confusion_matrix.png', dpi=100)
        plt.show()
        
        return f1, y_pred
    
    def predict_single_image(self, image_path):

        processed_img = self.preprocessor.preprocess_image(image_path)
        
        if processed_img is None:
            print(f"Error: Could not load image {image_path}")
            return None
        
        img_tensor = torch.FloatTensor(processed_img).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = self.model(img_tensor).cpu().numpy()[0][0]
        
        predicted_class = 'Flipping' if prediction > 0.5 else 'Not Flipping'
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        original_img = cv2.imread(image_path)
        if original_img is not None:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(8, 6))
            plt.imshow(original_img)
            plt.title(f'Prediction: {predicted_class}\nConfidence: {confidence:.2%}', 
                      fontsize=14, fontweight='bold')
            plt.axis('off')
            
            # Add color coding
            color = 'green' if predicted_class == 'Flipping' else 'red'
            plt.gca().patch.set_edgecolor(color)
            plt.gca().patch.set_linewidth(3)
            plt.show()
        
        return predicted_class, confidence
    
    def visualize_predictions(self, X_test, y_test, num_samples=8):

        num_samples = min(num_samples, len(X_test))
        
        X_test_tensor = torch.FloatTensor(X_test[:num_samples]).permute(0, 3, 1, 2).to(device)
        
        with torch.no_grad():
            predictions_prob = self.model(X_test_tensor).cpu().numpy()
        
        predictions = (predictions_prob > 0.5).astype(int).flatten()
        
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        axes = axes.ravel()
        
        for i in range(num_samples):
            # Display image
            axes[i].imshow(X_test[i].squeeze(), cmap='gray')
            
            # Set title with prediction info
            true_label = 'Flipping' if y_test[i] == 1 else 'Not Flipping'
            pred_label = 'Flipping' if predictions[i] == 1 else 'Not Flipping'
            confidence = predictions_prob[i][0] if predictions[i] == 1 else 1 - predictions_prob[i][0]
            
            color = 'green' if true_label == pred_label else 'red'
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2%}', 
                              fontsize=9, color=color, fontweight='bold')
            axes[i].axis('off')
        
        plt.suptitle('Model Predictions on Test Samples', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('figures/sample_predictions.png', dpi=100)
        plt.show()