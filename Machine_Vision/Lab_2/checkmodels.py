from google import genai

API_KEY = "AIzaSyCnOoXZjvVx1NKEGgB_R_sISz-dHphnUSM"
client = genai.Client(api_key=API_KEY)

print("Available models:")
print("="*60)

try:
    models = client.models.list()
    for model in models:
        # Check if it supports vision/content generation
        print(f"Model: {model.name}")
        if hasattr(model, 'supported_generation_methods'):
            print(f"  Methods: {model.supported_generation_methods}")
        print()
except Exception as e:
    print(f"Error listing models: {e}")