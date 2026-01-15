import cv2
import base64
from google import genai
import os

API_KEY = "AIzaSyCdgjf637X55HjPy4OUx5pSBiN5EVhs0Ug"

class CameraGeminiAnalyzer:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)

    def capture_image(self, output_path="capture.jpg"):
        print("📷 Opening USB camera...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise RuntimeError("Could not open camera")
	
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError("Failed to capture image")

        cv2.imwrite(output_path, frame)
        print(f"✅ Image saved: {output_path}")
        return output_path

    def analyze_image(self, image_path):
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        prompt = """
        Describe this image in short with one sentence.
        """

        response = self.client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents={
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_b64
                        }
                    }
                ]
            }
        )

        return response.text


if __name__ == "__main__":
    analyzer = CameraGeminiAnalyzer(API_KEY)

    image_path = analyzer.capture_image("usb_capture.jpg")
    description = analyzer.analyze_image(image_path)

    print("\n🧠 GEMINI IMAGE DESCRIPTION")
    print("=" * 60)
    print(description)
    print("=" * 60)
