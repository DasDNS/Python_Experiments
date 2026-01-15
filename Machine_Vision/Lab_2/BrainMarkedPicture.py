import google.genai as genai
import os
import cv2
import base64

class BrainTumorAnalyzer:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        
    def preprocess_image(self, image_path):
        """Preprocess MRI image"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # Save
        base, ext = os.path.splitext(image_path)
        processed_path = f"{base}_processed{ext}"
        cv2.imwrite(processed_path, denoised)
        
        return processed_path
    
    def mark_cancer_regions(self, image_path):
        """Draw markers on the processed image (placeholder example)"""
        img = cv2.imread(image_path)
        if img is None:
            return
        
        # Simulate marking cancerous regions
        # Replace these coordinates with actual tumor coordinates if available
        height, width = img.shape[:2]
        top_left = (int(width*0.3), int(height*0.3))
        bottom_right = (int(width*0.6), int(height*0.6))
        
        cv2.rectangle(img, top_left, bottom_right, (0,0,255), 2)  # red rectangle
        
        marked_path = image_path.replace("_processed", "_marked")
        cv2.imwrite(marked_path, img)
        return marked_path
    
    def analyze_mri(self, image_path, preprocess=True):
        """Analyze MRI image for tumor/cancer detection and suggest treatment plan"""
        try:
            if preprocess:
                print("Preprocessing image...")
                image_to_analyze = self.preprocess_image(image_path)
            else:
                image_to_analyze = image_path
            
            print(f"Reading: {image_to_analyze}")
            
            # Read image bytes
            with open(image_to_analyze, 'rb') as f:
                image_bytes = f.read()
            
            # Convert to base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Detect mime type
            ext = os.path.splitext(image_to_analyze)[1].lower()
            mime_type = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'
            
            print("Sending to Gemini API...")
            
            prompt = """
            Analyze this brain MRI scan and provide a detailed report:

            1. **Image Quality & Type**: Assess clarity and modality
            2. **Anatomical Assessment**: Evaluate brain structures
            3. **Cancer Detection**: Identify any tumors or cancerous regions
            4. **Location & Characteristics**: Describe findings in detail (size, shape, intensity)
            5. **Clinical Impression**: Severity, likely diagnosis, and cancer grade if applicable
            6. **Treatment Plan**: Suggest appropriate medical steps, therapies, or surgeries
            

            MEDICAL DISCLAIMER: This is an AI screening tool for educational purposes.
            Always consult qualified radiologists, oncologists, and neurologists for diagnosis and treatment.
            """
            
            response = self.client.models.generate_content(
                model='models/gemma-3-1b-it',
                contents={
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_b64
                            }
                        }
                    ]
                }
            )
            
            # Mark cancerous regions
            marked_image_path = self.mark_cancer_regions(image_to_analyze)
            
            return {
                'success': True,
                'original_image': image_path,
                'processed_image': image_to_analyze if preprocess else None,
                'marked_image': marked_image_path,
                'analysis': response.text
            }
            
        except Exception as e:
            import traceback
            return {
                'success': False,
                'error': f"{type(e).__name__}: {str(e)}",
                'details': traceback.format_exc()
            }


if __name__ == "__main__":
    API_KEY = "AIzaSyCnOoXZjvVx1NKEGgB_R_sISz-dHphnUSM"
    
    analyzer = BrainTumorAnalyzer(API_KEY)
    
    print("🧠 Brain Tumor & Cancer MRI Analyzer")
    print("="*60)
    
    result = analyzer.analyze_mri("Tr-gl_0010.jpg", preprocess=True)
    
    if result['success']:
        print("\n✅ ANALYSIS REPORT")
        print("="*60)
        print(result['analysis'])
        print("="*60)
        print(f"\n📁 Files:")
        print(f"   Original: {result['original_image']}")
        if result['processed_image']:
            print(f"   Enhanced: {result['processed_image']}")
        if result.get('marked_image'):
            print(f"   Marked Cancer Regions: {result['marked_image']}")
    else:
        print(f"\n❌ ERROR: {result['error']}")
        print(f"\n{result['details']}")
