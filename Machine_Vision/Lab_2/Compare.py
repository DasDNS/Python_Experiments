from google import genai
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
        
        # Save processed image
        base, ext = os.path.splitext(image_path)
        processed_path = f"{base}_processed{ext}"
        cv2.imwrite(processed_path, denoised)
        
        return processed_path
    
    def image_to_base64(self, image_path):
        """Convert image to base64 with mime type"""
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        ext = os.path.splitext(image_path)[1].lower()
        mime_type = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'
        
        return image_b64, mime_type
    
    def compare_mri_images(self, image1_path, image2_path, preprocess=True):
        """Compare two MRI images for tumor presence"""
        try:
            # Preprocess images
            if preprocess:
                print("Preprocessing images...")
                img1 = self.preprocess_image(image1_path)
                img2 = self.preprocess_image(image2_path)
            else:
                img1 = image1_path
                img2 = image2_path
            
            print("Preparing images for analysis...")
            
            img1_b64, img1_mime = self.image_to_base64(img1)
            img2_b64, img2_mime = self.image_to_base64(img2)
            
            prompt = """
            You are given TWO brain MRI scans: Image A and Image B.

            Perform a detailed comparative analysis and answer clearly:

            1. Image Quality & Orientation
            2. Tumor Presence:
               - Does Image A show a brain tumor? (Yes / No / Uncertain)
               - Does Image B show a brain tumor? (Yes / No / Uncertain)
            3. If tumor present:
               - Location
               - Size (relative)
               - Shape and intensity
            4. Comparison:
               - Which image shows more severe abnormality?
            5. Final Conclusion:
               - Image A: Tumor / No Tumor
               - Image B: Tumor / No Tumor

            MEDICAL DISCLAIMER:
            This is an AI-based screening comparison for educational purposes only.
            Always consult a certified radiologist for medical diagnosis.
            """
            
            print("Sending comparison request to Gemini API...")
            
            response = self.client.models.generate_content(
                model='models/gemini-2.5-flash',
                contents={
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": img1_mime,
                                "data": img1_b64
                            }
                        },
                        {
                            "inline_data": {
                                "mime_type": img2_mime,
                                "data": img2_b64
                            }
                        }
                    ]
                }
            )
            
            return {
                'success': True,
                'image_A_original': image1_path,
                'image_B_original': image2_path,
                'image_A_processed': img1 if preprocess else None,
                'image_B_processed': img2 if preprocess else None,
                'comparison_report': response.text
            }
        
        except Exception as e:
            import traceback
            return {
                'success': False,
                'error': f"{type(e).__name__}: {str(e)}",
                'details': traceback.format_exc()
            }


if __name__ == "__main__":
    API_KEY = "AIzaSyCdgjf637X55HjPy4OUx5pSBiN5EVhs0Ug"
    
    analyzer = BrainTumorAnalyzer(API_KEY)
    
    print("🧠 Brain MRI Tumor Comparison Tool")
    print("=" * 60)
    
    result = analyzer.compare_mri_images(
        "Tumour.jpg",
        "NoTumour.jpg",
        preprocess=True
    )
    
    if result['success']:
        print("\n✅ COMPARISON REPORT")
        print("=" * 60)
        print(result['comparison_report'])
        print("=" * 60)
        
        print("\n📁 Files:")
        print(f"   Image A Original : {result['image_A_original']}")
        print(f"   Image B Original : {result['image_B_original']}")
        if result['image_A_processed']:
            print(f"   Image A Enhanced : {result['image_A_processed']}")
            print(f"   Image B Enhanced : {result['image_B_processed']}")
    else:
        print(f"\n❌ ERROR: {result['error']}")
        print(result['details'])