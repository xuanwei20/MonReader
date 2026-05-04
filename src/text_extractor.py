import pytesseract
from gtts import gTTS
import playsound3
import os
import tempfile
from PIL import Image as PILImage
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from deep_translator import GoogleTranslator
import time

class TextExtractorAndReader:
    
    def __init__(self, tesseract_path=None, source_lang='tr', target_lang='en'):
        """
        Initialize TextExtractorAndReader with translation capabilities
        
        Args:
            tesseract_path: Path to Tesseract executable
            source_lang: Source language code (default: 'tr' for Turkish)
            target_lang: Target language code (default: 'en' for English)
        """
        # Language code mapping for deep_translator
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # Initialize translator with proper error handling
        try:
            self.translator = GoogleTranslator(source=source_lang, target=target_lang)
            print(f"Translator initialized: {source_lang} → {target_lang}")
        except Exception as e:
            print(f"Warning: Could not initialize translator: {e}")
            print("Will proceed without translation capability")
            self.translator = None
        
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        elif os.name == 'nt':  # Windows
            possible_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    print(f"Found Tesseract at: {path}")
                    break
        
    def extract_text_from_image(self, image_path, preprocess=True):
        """Extract text from image using OCR"""
        print(f"\nExtracting text from: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None
        
        if preprocess:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Noise removal
            denoised = cv2.medianBlur(gray, 3)
            
            # Thresholding
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Dilation to connect text components
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            dilated = cv2.dilate(thresh, kernel, iterations=1)
            
            processed_image = dilated
        else:
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Extract text using Tesseract
        try:
            # Configure Tesseract for Turkish and English text
            custom_config = r'--oem 3 --psm 6 -l tur+eng'
            extracted_text = pytesseract.image_to_string(processed_image, config=custom_config)
            
            # Remove extra whitespace and empty lines
            extracted_text = '\n'.join([line.strip() for line in extracted_text.split('\n') if line.strip()])
            
            if extracted_text:
                print(f"Text extracted successfully!")
                print(f"Extracted text ({len(extracted_text)} characters):")
                print("-" * 50)
                print(extracted_text[:500])
                if len(extracted_text) > 500:
                    print(f"... and {len(extracted_text) - 500} more characters")
                print("-" * 50)
            else:
                print("No text found in the image")
                
            return extracted_text
            
        except Exception as e:
            print(f"Error during OCR: {str(e)}")
            print("Make sure Tesseract is installed on your system")
            print("For Turkish text, install Turkish language pack: tesseract-lang-tur")
            return None
    
    def translate_text(self, text, show_translation=True):
        """Translate text from source language to target language"""
        if not text or len(text.strip()) == 0:
            print("No text to translate")
            return None
        
        if self.translator is None:
            print("Translator not available. Please install deep_translator and check language codes.")
            return text
        
        print(f"\nTranslating text from {self.source_lang} to {self.target_lang}...")
        
        try:
            # Split long text into chunks (Google Translate has character limits)
            max_chunk_size = 5000
            chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            
            translated_chunks = []
            for i, chunk in enumerate(chunks):
                print(f"Translating chunk {i+1}/{len(chunks)}...")
                try:
                    translated_chunk = self.translator.translate(chunk)
                except AttributeError:
                    translated_chunk = self.translator.translate(text=chunk)
                translated_chunks.append(translated_chunk)
            
            translated_text = ' '.join(translated_chunks)
            
            if show_translation and translated_text:
                print("\n" + "="*50)
                print(f"TRANSLATED TEXT ({self.target_lang.upper()}):")
                print("="*50)
                print(translated_text[:500])
                if len(translated_text) > 500:
                    print(f"... and {len(translated_text) - 500} more characters")
                print("="*50)
            
            return translated_text
            
        except Exception as e:
            print(f"Error during translation: {str(e)}")
            print("Using original text for speech synthesis")
            return text
    
    def save_audio_file(self, text, output_path, language='en', slow=False):
        """
        Save text as audio file without playing it
        
        Args:
            text: Text to convert to speech
            output_path: Path to save the audio file (should end with .mp3)
            language: Language code for speech
            slow: Whether to speak slowly
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not text or len(text.strip()) == 0:
            print("No text to convert to audio")
            return False
        
        try:
            # Create gTTS object
            tts = gTTS(text=text, lang=language, slow=slow)
            
            # Save to specified path
            tts.save(output_path)
            print(f"Audio saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error saving audio file: {str(e)}")
            return False
    
    def text_to_speech(self, text, language='en', slow=False, save_audio=False, audio_output_path=None):
        """
        Convert text to speech and optionally save audio file
        
        Args:
            text: Text to convert to speech
            language: Language code for speech
            slow: Whether to speak slowly
            save_audio: Whether to save audio to file
            audio_output_path: Path to save audio file (if save_audio is True)
        
        Returns:
            Path to saved audio file if save_audio is True, else None
        """
        if not text or len(text.strip()) == 0:
            print("No text to convert to speech")
            return None
        
        print(f"\nConverting text to speech (Language: {language})...")
        print(f"Text to read ({len(text)} characters):")
        print("-" * 50)
        preview = text[:200] + "..." if len(text) > 200 else text
        print(preview)
        print("-" * 50)
        
        saved_audio_path = None
        
        try:
            # Create gTTS object
            tts = gTTS(text=text, lang=language, slow=slow)
            
            # Save audio if requested
            if save_audio:
                if audio_output_path is None:
                    # Generate default filename with timestamp
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    audio_output_path = f"speech_output_{timestamp}.mp3"
                
                tts.save(audio_output_path)
                print(f"Audio saved to: {audio_output_path}")
                saved_audio_path = audio_output_path
            
            # Play the audio using temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_filename = temp_file.name
                tts.save(temp_filename)
                
                print("Playing audio...")
                playsound3.playsound(temp_filename)
                
                # Clean up temporary file
                os.unlink(temp_filename)
                
            print("Audio playback complete.")
            
            return saved_audio_path
            
        except Exception as e:
            print(f"Error during text-to-speech: {str(e)}")
            return None
    
    def process_image_with_audio(self, image_path, preprocess=True, 
                                 translate=True, target_lang='en', 
                                 slow=False, save_translated=True,
                                 save_audio=True, audio_output_dir=None):
        """
        Process image: extract text, translate (optional), convert to speech, and save audio
        
        Args:
            image_path: Path to image file
            preprocess: Apply preprocessing for better OCR
            translate: Whether to translate text before speech
            target_lang: Target language for speech (default: 'en')
            slow: Speak slowly
            save_translated: Save translated text to file
            save_audio: Whether to save audio file
            audio_output_dir: Directory to save audio files (if None, uses current directory)
        
        Returns:
            Dictionary containing extracted text, translated text, and audio file path
        """
        print("\n" + "="*60)
        print("TEXT EXTRACTION, TRANSLATION AND AUDIO FEATURE")
        print("="*60)
        
        result = {
            'original_text': None,
            'translated_text': None,
            'audio_file': None,
            'text_file': None
        }
        
        # Step 1: Extract text
        extracted_text = self.extract_text_from_image(image_path, preprocess)
        
        if extracted_text and len(extracted_text) > 0:
            result['original_text'] = extracted_text
            
            # Step 2: Translate if requested
            if translate and self.translator is not None:
                translated_text = self.translate_text(extracted_text, show_translation=True)
                text_for_speech = translated_text if translated_text else extracted_text
                speech_language = self.target_lang
                result['translated_text'] = translated_text
            else:
                if translate and self.translator is None:
                    print("Translation requested but translator not available. Using original text.")
                text_for_speech = extracted_text
                speech_language = self.source_lang
                print("\nSkipping translation, using original text")
            
            # Prepare audio output path
            audio_file_path = None
            if save_audio:
                if audio_output_dir:
                    os.makedirs(audio_output_dir, exist_ok=True)
                    base_name = Path(image_path).stem
                    audio_file_path = os.path.join(audio_output_dir, f"{base_name}_speech.mp3")
                else:
                    base_name = Path(image_path).stem
                    audio_file_path = f"{base_name}_speech_{self.target_lang}.mp3"
            
            # Step 3: Convert to speech and optionally save audio
            saved_audio = self.text_to_speech(
                text_for_speech, 
                language=speech_language, 
                slow=slow,
                save_audio=save_audio,
                audio_output_path=audio_file_path
            )
            
            if saved_audio:
                result['audio_file'] = saved_audio
            
            # Step 4: Save texts to file
            base_name = Path(image_path).stem
            output_file = f"extracted_text_{base_name}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Text extracted from: {image_path}\n")
                f.write(f"Audio file: {result['audio_file'] if result['audio_file'] else 'Not saved'}\n")
                f.write("="*50 + "\n")
                f.write("ORIGINAL TEXT:\n")
                f.write("="*50 + "\n")
                f.write(extracted_text)
                if result['translated_text']:
                    f.write("\n\n" + "="*50 + "\n")
                    f.write(f"TRANSLATED TEXT ({self.target_lang.upper()}):\n")
                    f.write("="*50 + "\n")
                    f.write(result['translated_text'])
            print(f"\nText saved to: {output_file}")
            result['text_file'] = output_file
            
            # Print summary
            print("\n" + "="*60)
            print("PROCESSING COMPLETE - SUMMARY")
            print("="*60)
            print(f"Original text extracted: {len(extracted_text)} characters")
            if result['translated_text']:
                print(f"Text translated to {self.target_lang.upper()}: {len(result['translated_text'])} characters")
            if result['audio_file']:
                print(f"Audio saved to: {result['audio_file']}")
            print(f"Text saved to: {result['text_file']}")
            print("="*60)
            
            return result
        else:
            print("No text found or extraction failed")
            return None
    
    def visualize_ocr_result(self, image_path):
        """Visualize OCR results with bounding boxes"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get detailed OCR data
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            vis_image = rgb_image.copy()
            
            # Draw bounding boxes and text
            n_boxes = len(data['text'])
            text_detected = False
            
            for i in range(n_boxes):
                if int(data['conf'][i]) > 30:  # Only show confident detections
                    text_detected = True
                    (x, y, w, h) = (data['left'][i], data['top'][i], 
                                   data['width'][i], data['height'][i])
                    # Draw rectangle
                    cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # Put text
                    text = data['text'][i].strip()
                    if text:
                        cv2.putText(vis_image, text, (x, y - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if text_detected:
                plt.figure(figsize=(12, 8))
                plt.imshow(vis_image)
                plt.title('OCR Detection Results - Text Regions Highlighted', 
                         fontsize=14, fontweight='bold')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig('figures/ocr_visualization.png', dpi=100)
                plt.show()
                print("OCR visualization saved as 'ocr_visualization.png'")
            else:
                print("No text regions detected in the image")
                
        except Exception as e:
            print(f"Error in OCR visualization: {str(e)}")


def test_ocr_on_image(image_path, tesseract_path=None, source_lang='tr', target_lang='en'):
    """Test function for OCR with translation and audio saving"""
    text_reader = TextExtractorAndReader(
        tesseract_path=tesseract_path,
        source_lang=source_lang,
        target_lang=target_lang
    )
    
    result = text_reader.process_image_with_audio(
        image_path, 
        preprocess=True, 
        translate=True,
        target_lang=target_lang,
        slow=False,
        save_translated=True,
        save_audio=True,  # Save audio file
        audio_output_dir="audio_outputs"  # Create a directory for audio files
    )
    
    text_reader.visualize_ocr_result(image_path)
    
    return result